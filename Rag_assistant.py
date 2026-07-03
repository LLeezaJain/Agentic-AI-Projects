from __future__ import annotations

import argparse
import os
import threading
from pathlib import Path
from typing import Iterable, Literal
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


class AssistantError(RuntimeError):
    pass


class ConfigurationError(AssistantError):
    pass


class UnsupportedDocumentTypeError(AssistantError):
    pass


class EmptyCorpusError(AssistantError):
    pass


class UserNotFoundError(AssistantError):
    pass


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    groq_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("GROQ_API_KEY", "")),
        validate_default=True,
    )
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.2
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    persist_directory: Path = Path("./chroma_db")
    collection_name: str = "production_rag"
    chunk_size: int = Field(default=1000, ge=128, le=8000)
    chunk_overlap: int = Field(default=150, ge=0, le=2000)
    top_k: int = Field(default=5, ge=1, le=25)

    @field_validator("groq_api_key")
    @classmethod
    def require_groq_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ConfigurationError("GROQ_API_KEY is required.")
        return value

    @field_validator("persist_directory")
    @classmethod
    def normalize_persist_directory(cls, value: Path) -> Path:
        return value.expanduser().resolve()


class SourceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    page: int | None = None
    chunk_id: str | None = None


class ChatTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["human", "ai"]
    content: str = Field(min_length=1)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=25)


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str
    question: str
    answer: str
    sources: list[SourceDocument]


class BatchQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    queries: list[QueryRequest] = Field(min_length=1, max_length=100)


class BatchQueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    responses: list[QueryResponse]


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    paths: list[Path] = Field(min_length=1)

    @field_validator("paths")
    @classmethod
    def normalize_paths(cls, values: list[Path]) -> list[Path]:
        normalized = [path.expanduser().resolve() for path in values]
        missing = [str(path) for path in normalized if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Input path(s) do not exist: {missing}")
        return normalized


class IngestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    indexed_documents: int
    indexed_chunks: int
    collection_name: str
    persist_directory: str


class UserSession(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    user_id: str
    history: list[BaseMessage] = Field(default_factory=list)
    lock: threading.RLock = Field(default_factory=threading.RLock)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, UserSession] = {}
        self._lock = threading.RLock()

    def get_or_create(self, user_id: str) -> UserSession:
        if not user_id.strip():
            raise ValueError("user_id cannot be empty.")

        with self._lock:
            session = self._sessions.get(user_id)
            if session is None:
                session = UserSession(user_id=user_id)
                self._sessions[user_id] = session
            return session

    def reset(self, user_id: str) -> None:
        with self._lock:
            if user_id not in self._sessions:
                raise UserNotFoundError(f"Unknown user_id: {user_id}")
            self._sessions[user_id].history.clear()

    def delete(self, user_id: str) -> None:
        with self._lock:
            if user_id not in self._sessions:
                raise UserNotFoundError(f"Unknown user_id: {user_id}")
            del self._sessions[user_id]


class ProductionRagAssistant:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._vectorstore_lock = threading.RLock()
        self._sessions = SessionStore()

        self._embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )

        self._vectorstore = Chroma(
            collection_name=config.collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(config.persist_directory),
        )

        self._llm = ChatGroq(
            model=config.groq_model,
            temperature=config.groq_temperature,
            api_key=config.groq_api_key.get_secret_value(),
        )

        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a production RAG assistant. Answer only from the supplied context. "
                    "If the answer is not present in the context, say you do not know. "
                    "Cite source names naturally when useful.\n\nContext:\n{context}",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        self._chain: Runnable = self._prompt | self._llm

    def ingest(self, request: IngestRequest) -> IngestResponse:
        documents = self._load_documents(request.paths)
        if not documents:
            raise EmptyCorpusError("No documents were loaded.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        if not chunks:
            raise EmptyCorpusError("Documents were loaded but produced zero chunks.")

        ids = [str(uuid4()) for _ in chunks]
        for index, chunk in enumerate(chunks):
            chunk.metadata = dict(chunk.metadata)
            chunk.metadata["chunk_id"] = ids[index]

        with self._vectorstore_lock:
            self._vectorstore.add_documents(chunks, ids=ids)
            persist = getattr(self._vectorstore, "persist", None)
            if callable(persist):
                persist()

        return IngestResponse(
            indexed_documents=len(documents),
            indexed_chunks=len(chunks),
            collection_name=self.config.collection_name,
            persist_directory=str(self.config.persist_directory),
        )

    def query(self, request: QueryRequest) -> QueryResponse:
        session = self._sessions.get_or_create(request.user_id)
        top_k = request.top_k or self.config.top_k

        with session.lock:
            docs = self._retrieve(request.question, top_k)
            context = self._format_context(docs)

            message = self._chain.invoke(
                {
                    "context": context,
                    "history": list(session.history),
                    "question": request.question,
                }
            )

            answer = self._message_content(message)
            session.history.append(self._human_message(request.question))
            session.history.append(self._ai_message(answer))

            return QueryResponse(
                user_id=request.user_id,
                question=request.question,
                answer=answer,
                sources=self._sources_from_documents(docs),
            )

    def batch_query(self, request: BatchQueryRequest) -> BatchQueryResponse:
        responses = [self.query(query) for query in request.queries]
        return BatchQueryResponse(responses=responses)

    def reset_user(self, user_id: str) -> None:
        self._sessions.reset(user_id)

    def delete_user(self, user_id: str) -> None:
        self._sessions.delete(user_id)

    def _retrieve(self, question: str, top_k: int) -> list[Document]:
        with self._vectorstore_lock:
            docs = self._vectorstore.similarity_search(question, k=top_k)

        if not docs:
            raise EmptyCorpusError(
                "The vector store returned no documents. Ingest source files before querying."
            )
        return docs

    @staticmethod
    def _format_context(documents: Iterable[Document]) -> str:
        formatted: list[str] = []
        for index, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            location = f"{source}, page {page}" if page is not None else str(source)
            formatted.append(f"[{index}] Source: {location}\n{doc.page_content}")
        return "\n\n".join(formatted)

    @staticmethod
    def _sources_from_documents(documents: Iterable[Document]) -> list[SourceDocument]:
        sources: list[SourceDocument] = []
        seen: set[tuple[str, int | None, str | None]] = set()

        for doc in documents:
            source = str(doc.metadata.get("source", "unknown"))
            page_value = doc.metadata.get("page")
            page = int(page_value) if isinstance(page_value, int) or str(page_value).isdigit() else None
            chunk_id = doc.metadata.get("chunk_id")
            chunk = str(chunk_id) if chunk_id is not None else None
            key = (source, page, chunk)

            if key not in seen:
                sources.append(SourceDocument(source=source, page=page, chunk_id=chunk))
                seen.add(key)

        return sources

    @staticmethod
    def _message_content(message: object) -> str:
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        raise AssistantError("LLM returned an empty or unsupported response payload.")

    @staticmethod
    def _human_message(content: str) -> BaseMessage:
        from langchain_core.messages import HumanMessage

        return HumanMessage(content=content)

    @staticmethod
    def _ai_message(content: str) -> BaseMessage:
        from langchain_core.messages import AIMessage

        return AIMessage(content=content)

    @classmethod
    def _load_documents(cls, paths: Iterable[Path]) -> list[Document]:
        documents: list[Document] = []
        for path in paths:
            if path.is_dir():
                documents.extend(cls._load_documents(sorted(file for file in path.rglob("*") if file.is_file())))
            else:
                documents.extend(cls._load_file(path))
        return documents

    @staticmethod
    def _load_file(path: Path) -> list[Document]:
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif suffix == ".csv":
            loader = CSVLoader(str(path))
        elif suffix in {".md", ".markdown"}:
            loader = UnstructuredMarkdownLoader(str(path))
        elif suffix in {".txt", ".log", ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".json", ".yaml", ".yml"}:
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            raise UnsupportedDocumentTypeError(f"Unsupported file type for {path}")

        documents = loader.load()
        for doc in documents:
            doc.metadata = dict(doc.metadata)
            doc.metadata["source"] = str(path)
        return documents


def build_assistant() -> ProductionRagAssistant:
    return ProductionRagAssistant(AppConfig())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production RAG assistant using ChatGroq and HuggingFace embeddings.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("paths", nargs="+", type=Path)

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--user-id", default="default")
    ask_parser.add_argument("--top-k", type=int, default=None)

    batch_parser = subparsers.add_parser("batch")
    batch_parser.add_argument("questions", nargs="+")
    batch_parser.add_argument("--user-id", default="default")
    batch_parser.add_argument("--top-k", type=int, default=None)

    reset_parser = subparsers.add_parser("reset-user")
    reset_parser.add_argument("--user-id", required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assistant = build_assistant()

    if args.command == "ingest":
        response = assistant.ingest(IngestRequest(paths=args.paths))
        print(response.model_dump_json(indent=2))
        return

    if args.command == "ask":
        response = assistant.query(
            QueryRequest(user_id=args.user_id, question=args.question, top_k=args.top_k)
        )
        print(response.model_dump_json(indent=2))
        return

    if args.command == "batch":
        request = BatchQueryRequest(
            queries=[
                QueryRequest(user_id=args.user_id, question=question, top_k=args.top_k)
                for question in args.questions
            ]
        )
        response = assistant.batch_query(request)
        print(response.model_dump_json(indent=2))
        return

    if args.command == "reset-user":
        assistant.reset_user(args.user_id)
        print('{"status":"ok"}')
        return

    raise AssistantError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()

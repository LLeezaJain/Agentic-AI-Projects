from __future__ import annotations
import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

load_dotenv()


class ChatWithPDFError(RuntimeError):
    pass


class ConfigurationError(ChatWithPDFError):
    pass


class PdfExtractionError(ChatWithPDFError):
    pass


class EmptyDocumentError(ChatWithPDFError):
    pass


class UnsupportedVectorStoreError(ChatWithPDFError):
    pass


class SessionNotFoundError(ChatWithPDFError):
    pass


VectorStoreKind = Literal["faiss", "chroma"]


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    groq_api_key: SecretStr = Field(default_factory=lambda: SecretStr(os.getenv("GROQ_API_KEY", "")))
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size_words: int = Field(default=800, ge=100, le=3000)
    chunk_overlap_words: int = Field(default=150, ge=0, le=1000)
    top_k: int = Field(default=4, ge=1, le=25)
    batch_workers: int = Field(default=4, ge=1, le=32)
    chroma_persist_directory: Path = Path("./chroma_pdf_db")
    faiss_persist_directory: Path = Path("./faiss_pdf_db")

    @field_validator("groq_api_key")
    @classmethod
    def require_groq_api_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ConfigurationError("GROQ_API_KEY is required.")
        return value

    @field_validator("chroma_persist_directory", "faiss_persist_directory")
    @classmethod
    def normalize_paths(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @field_validator("chunk_overlap_words")
    @classmethod
    def validate_overlap(cls, value: int, info: Any) -> int:
        chunk_size = info.data.get("chunk_size_words")
        if isinstance(chunk_size, int) and value >= chunk_size:
            raise ValueError("chunk_overlap_words must be smaller than chunk_size_words.")
        return value


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(min_length=1)
    page: int = Field(ge=1)
    chunk_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1, max_length=128)


class DocumentChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str = Field(min_length=1)
    metadata: DocumentMetadata


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str = Field(min_length=1)
    metadata: DocumentMetadata
    score: float
    backend: str = Field(min_length=1)


class UploadPDFRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str = Field(default="default", min_length=1, max_length=128)
    pdf_path: Path

    @field_validator("pdf_path")
    @classmethod
    def validate_pdf_path(cls, value: Path) -> Path:
        resolved = value.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"PDF not found: {resolved}")
        if not resolved.is_file():
            raise PdfExtractionError(f"Path is not a file: {resolved}")
        if resolved.suffix.lower() != ".pdf":
            raise PdfExtractionError(f"Only PDF files are supported: {resolved}")
        return resolved


class UploadPDFResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str
    status: Literal["indexed"]
    document: str
    chunks: int = Field(ge=0)
    backend: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str = Field(default="default", min_length=1, max_length=128)
    question: str = Field(min_length=1)
    filter_doc: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=25)


class SourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    page: int
    chunk_id: str
    score: float


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str
    question: str
    answer: str
    sources: list[SourceReference]
    chunks_used: int = Field(ge=0)
    avg_relevance_score: float = Field(ge=0.0)
    backend: str


class BatchChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requests: list[ChatRequest] = Field(min_length=1, max_length=100)


class BatchChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    responses: list[ChatResponse]


class BackendProtocol(Protocol):
    backend_name: str

    def add_documents(self, documents: list[DocumentChunk]) -> int:
        ...

    def search(self, query: str, k: int, filter_doc: str | None = None) -> list[RetrievedChunk]:
        ...


class FAISSBackend:
    backend_name = "FAISS"

    def __init__(self, embedding_function: HuggingFaceEmbeddings, persist_directory: Path) -> None:
        self._embedding_function = embedding_function
        self._persist_directory = persist_directory
        self._store: Any | None = None
        self._lock = threading.RLock()
        self._load_store()

    def _load_store(self) -> None:
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError:
            return

        with self._lock:
            if self._persist_directory.exists() and (self._persist_directory / "index.faiss").exists():
                self._store = FAISS.load_local(
                    str(self._persist_directory),
                    self._embedding_function,
                    allow_dangerous_deserialization=True,
                )

    def add_documents(self, documents: list[DocumentChunk]) -> int:
        if not documents:
            return 0

        try:
            from langchain_community.vectorstores import FAISS
        except ImportError as exc:
            raise ConfigurationError("Install langchain-community and faiss-cpu to use FAISS.") from exc

        langchain_docs = [self._to_langchain_document(chunk) for chunk in documents]

        with self._lock:
            if self._store is None:
                self._store = FAISS.from_documents(langchain_docs, self._embedding_function)
            else:
                self._store.add_documents(langchain_docs)
            
            self._persist_directory.mkdir(parents=True, exist_ok=True)
            self._store.save_local(str(self._persist_directory))

        return len(documents)

    def search(self, query: str, k: int, filter_doc: str | None = None) -> list[RetrievedChunk]:
        with self._lock:
            if self._store is None:
                raise EmptyDocumentError("No documents have been indexed for this session.")
            results = self._store.similarity_search_with_score(query, k=max(k * 4, k))

        chunks = [self._from_langchain_result(doc, score) for doc, score in results]
        if filter_doc is not None:
            chunks = [chunk for chunk in chunks if chunk.metadata.source == filter_doc]
        return chunks[:k]

    @staticmethod
    def _to_langchain_document(chunk: DocumentChunk) -> Document:
        return Document(
            page_content=chunk.content,
            metadata=chunk.metadata.model_dump(),
        )

    def _from_langchain_result(self, doc: Document, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            content=doc.page_content,
            metadata=DocumentMetadata.model_validate(doc.metadata),
            score=float(score),
            backend=self.backend_name,
        )


class ChromaDBBackend:
    backend_name = "ChromaDB"

    def __init__(
        self,
        embedding_function: HuggingFaceEmbeddings,
        collection_name: str,
        persist_directory: Path,
    ) -> None:
        self._embedding_function = embedding_function
        self._lock = threading.RLock()

        try:
            from langchain_chroma import Chroma
        except ImportError as exc:
            raise ConfigurationError("Install langchain-chroma and chromadb to use ChromaDB.") from exc

        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=str(persist_directory),
        )

    def add_documents(self, documents: list[DocumentChunk]) -> int:
        if not documents:
            return 0

        langchain_docs = [
            Document(page_content=chunk.content, metadata=chunk.metadata.model_dump())
            for chunk in documents
        ]
        ids = [chunk.metadata.chunk_id for chunk in documents]

        with self._lock:
            self._store.add_documents(langchain_docs, ids=ids)
            persist = getattr(self._store, "persist", None)
            if callable(persist):
                persist()

        return len(documents)

    def search(self, query: str, k: int, filter_doc: str | None = None) -> list[RetrievedChunk]:
        where = {"source": filter_doc} if filter_doc else None

        with self._lock:
            results = self._store.similarity_search_with_score(query, k=k, filter=where)

        return [
            RetrievedChunk(
                content=doc.page_content,
                metadata=DocumentMetadata.model_validate(doc.metadata),
                score=float(score),
                backend=self.backend_name,
            )
            for doc, score in results
        ]


class PDFProcessor:
    def __init__(self, chunk_size_words: int, chunk_overlap_words: int) -> None:
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words

    def process(self, request: UploadPDFRequest) -> list[DocumentChunk]:
        try:
            import pypdf
        except ImportError as exc:
            raise ConfigurationError("Install pypdf to process PDFs: pip install pypdf") from exc

        chunks: list[DocumentChunk] = []
        source_name = request.pdf_path.name

        with request.pdf_path.open("rb") as file_handle:
            reader = pypdf.PdfReader(file_handle)
            if len(reader.pages) == 0:
                raise EmptyDocumentError(f"PDF contains no pages: {request.pdf_path}")

            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                if not page_text or not page_text.strip():
                    continue

                for chunk_index, chunk_text in enumerate(self._chunk_text(page_text), start=1):
                    chunk_id = f"{request.user_id}:{source_name}:p{page_number}:c{chunk_index}:{uuid4().hex}"
                    chunks.append(
                        DocumentChunk(
                            content=chunk_text,
                            metadata=DocumentMetadata(
                                source=source_name,
                                page=page_number,
                                chunk_id=chunk_id,
                                user_id=request.user_id,
                            ),
                        )
                    )

        if not chunks:
            raise EmptyDocumentError(f"No extractable text chunks found in PDF: {request.pdf_path}")

        return chunks

    def _chunk_text(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return []

        step = self.chunk_size_words - self.chunk_overlap_words
        chunks: list[str] = []

        for start in range(0, len(words), step):
            chunk_words = words[start : start + self.chunk_size_words]
            chunk = " ".join(chunk_words).strip()
            if len(chunk) >= 50:
                chunks.append(chunk)

        return chunks


class AnswerGenerator:
    PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert document analyst. Answer based only on the provided PDF context. "
                "If the answer is not available in the context, say exactly: "
                '"This information is not available in the uploaded document." '
                "Cite source pages when possible.",
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}\n\n"
                "Provide a precise, well-structured answer.",
            ),
        ]
    )

    def __init__(self, llm: ChatGroq) -> None:
        self._chain = self.PROMPT | llm
        self._lock = threading.RLock()

    def generate(self, request: ChatRequest, retrieved_chunks: list[RetrievedChunk]) -> ChatResponse:
        if not retrieved_chunks:
            return ChatResponse(
                user_id=request.user_id,
                question=request.question,
                answer="No relevant content found. Try rephrasing your question.",
                sources=[],
                chunks_used=0,
                avg_relevance_score=0.0,
                backend="none",
            )

        context = "\n\n".join(
            f"[Source: {chunk.metadata.source}, Page {chunk.metadata.page}, Chunk {chunk.metadata.chunk_id}]\n"
            f"{chunk.content}"
            for chunk in retrieved_chunks
        )

        with self._lock:
            response = self._chain.invoke({"context": context, "question": request.question})

        answer = self._extract_text(response)
        sources = self._source_references(retrieved_chunks)
        avg_score = sum(chunk.score for chunk in retrieved_chunks) / len(retrieved_chunks)

        return ChatResponse(
            user_id=request.user_id,
            question=request.question,
            answer=answer,
            sources=sources,
            chunks_used=len(retrieved_chunks),
            avg_relevance_score=avg_score,
            backend=retrieved_chunks[0].backend,
        )

    @staticmethod
    def _extract_text(response: object) -> str:
        content = getattr(response, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        raise ChatWithPDFError("LLM returned an empty or unsupported response.")

    @staticmethod
    def _source_references(chunks: list[RetrievedChunk]) -> list[SourceReference]:
        seen: set[tuple[str, int, str]] = set()
        sources: list[SourceReference] = []

        for chunk in chunks:
            key = (chunk.metadata.source, chunk.metadata.page, chunk.metadata.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                SourceReference(
                    source=chunk.metadata.source,
                    page=chunk.metadata.page,
                    chunk_id=chunk.metadata.chunk_id,
                    score=chunk.score,
                )
            )

        return sources


class UserSession(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    user_id: str
    backend: Any
    indexed_docs: set[str] = Field(default_factory=set)
    lock: Any = Field(default_factory=threading.RLock)


class SessionStore:
    def __init__(self, config: AppConfig, embeddings: HuggingFaceEmbeddings, vector_store: VectorStoreKind) -> None:
        self._config = config
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._sessions: dict[str, UserSession] = {}
        self._lock = threading.RLock()

    def get_or_create(self, user_id: str) -> UserSession:
        with self._lock:
            session = self._sessions.get(user_id)
            if session is None:
                session = UserSession(user_id=user_id, backend=self._new_backend(user_id))
                self._sessions[user_id] = session
            return session

    def delete(self, user_id: str) -> None:
        with self._lock:
            if user_id not in self._sessions:
                raise SessionNotFoundError(f"Unknown user_id: {user_id}")
            del self._sessions[user_id]

    def _new_backend(self, user_id: str) -> BackendProtocol:
        if self._vector_store == "faiss":
            safe_user = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in user_id)
            persist_dir = self._config.faiss_persist_directory / f"faiss_{safe_user}"
            return FAISSBackend(self._embeddings, persist_directory=persist_dir)

        if self._vector_store == "chroma":
            safe_user = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in user_id)
            collection_name = f"pdf_documents_{safe_user}"
            return ChromaDBBackend(
                embedding_function=self._embeddings,
                collection_name=collection_name,
                persist_directory=self._config.chroma_persist_directory,
            )

        raise UnsupportedVectorStoreError(f"Unsupported vector store: {self._vector_store}")


class ChatWithPDF:
    def __init__(self, config: AppConfig | None = None, vector_store: VectorStoreKind = "faiss") -> None:
        self.config = config or AppConfig()
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
        self._llm = ChatGroq(
            model=self.config.groq_model,
            temperature=self.config.groq_temperature,
            groq_api_key=self.config.groq_api_key.get_secret_value(),
        )
        self._processor = PDFProcessor(
            chunk_size_words=self.config.chunk_size_words,
            chunk_overlap_words=self.config.chunk_overlap_words,
        )
        self._generator = AnswerGenerator(self._llm)
        self._sessions = SessionStore(
            config=self.config,
            embeddings=self._embeddings,
            vector_store=vector_store,
        )

    def upload_pdf(self, request: UploadPDFRequest) -> UploadPDFResponse:
        session = self._sessions.get_or_create(request.user_id)

        with session.lock:
            chunks = self._processor.process(request)
            count = session.backend.add_documents(chunks)
            session.indexed_docs.add(request.pdf_path.name)

            return UploadPDFResponse(
                user_id=request.user_id,
                status="indexed",
                document=request.pdf_path.name,
                chunks=count,
                backend=session.backend.backend_name,
            )

    def chat(self, request: ChatRequest) -> ChatResponse:
        session = self._sessions.get_or_create(request.user_id)

        with session.lock:
            top_k = request.top_k or self.config.top_k
            retrieved = session.backend.search(
                query=request.question,
                k=top_k,
                filter_doc=request.filter_doc,
            )
            return self._generator.generate(request, retrieved)

    def chat_batch(self, request: BatchChatRequest) -> BatchChatResponse:
        max_workers = min(self.config.batch_workers, len(request.requests))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            responses = list(executor.map(self.chat, request.requests))
        return BatchChatResponse(responses=responses)

    def delete_user_session(self, user_id: str) -> None:
        self._sessions.delete(user_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with PDF using ChatGroq and HuggingFace embeddings.")
    parser.add_argument("--vector-store", choices=["faiss", "chroma"], default="faiss")

    subparsers = parser.add_subparsers(dest="command", required=True)

    upload_parser = subparsers.add_parser("upload")
    upload_parser.add_argument("pdf", type=Path)
    upload_parser.add_argument("--user-id", default="default")

    chat_parser = subparsers.add_parser("chat")
    chat_parser.add_argument("question")
    chat_parser.add_argument("--user-id", default="default")
    chat_parser.add_argument("--filter-doc", default=None)
    chat_parser.add_argument("--top-k", type=int, default=None)

    batch_parser = subparsers.add_parser("batch")
    batch_parser.add_argument("batch_json", type=Path)

    return parser.parse_args()


def _chat_request_from_mapping(payload: object) -> ChatRequest:
    if not isinstance(payload, dict):
        raise ChatWithPDFError("Each batch item must be a JSON object.")
    return ChatRequest.model_validate(payload)


def main() -> None:
    args = parse_args()
    system = ChatWithPDF(vector_store=args.vector_store)

    if args.command == "upload":
        response = system.upload_pdf(UploadPDFRequest(user_id=args.user_id, pdf_path=args.pdf))
        print(response.model_dump_json(indent=2))
        return

    if args.command == "chat":
        response = system.chat(
            ChatRequest(
                user_id=args.user_id,
                question=args.question,
                filter_doc=args.filter_doc,
                top_k=args.top_k,
            )
        )
        print(response.model_dump_json(indent=2))
        return

    if args.command == "batch":
        batch_path = args.batch_json.expanduser().resolve()
        with batch_path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)

        if not isinstance(payload, list):
            raise ChatWithPDFError("Batch JSON must contain a list of chat request objects.")

        response = system.chat_batch(
            BatchChatRequest(requests=[_chat_request_from_mapping(item) for item in payload])
        )
        print(response.model_dump_json(indent=2))
        return

    raise ChatWithPDFError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()

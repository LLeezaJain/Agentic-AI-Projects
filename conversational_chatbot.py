from __future__ import annotations
import argparse
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

load_dotenv()


class ChatbotError(RuntimeError):
    pass


class ConfigurationError(ChatbotError):
    pass


class SessionNotFoundError(ChatbotError):
    pass


class InvalidMessageError(ChatbotError):
    pass


MemoryStrategy = Literal["buffer", "window", "summary"]
MessageRole = Literal["system", "user", "assistant"]


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    groq_api_key: SecretStr = Field(default_factory=lambda: SecretStr(os.getenv("GROQ_API_KEY", "")))
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    default_memory_strategy: MemoryStrategy = "buffer"
    default_window_size: int = Field(default=10, ge=4, le=100)
    max_response_tokens: int = Field(default=500, ge=32, le=4096)
    batch_workers: int = Field(default=4, ge=1, le=32)

    @field_validator("groq_api_key")
    @classmethod
    def require_groq_api_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ConfigurationError("GROQ_API_KEY is required.")
        return value


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: MessageRole
    content: str = Field(min_length=1)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MemoryStats(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: MemoryStrategy
    stored_messages: int = Field(ge=0)
    has_summary: bool
    context_window: int = Field(ge=1)
    total_tokens_used: int = Field(ge=0)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str = Field(default="default", min_length=1, max_length=128)
    message: str = Field(min_length=1)
    memory_strategy: MemoryStrategy | None = None
    window_size: int | None = Field(default=None, ge=4, le=100)


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str
    message: ChatMessage
    reply: ChatMessage
    memory_stats: MemoryStats


class BatchChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requests: list[ChatRequest] = Field(min_length=1, max_length=100)


class BatchChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    responses: list[ChatResponse]


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str = Field(min_length=1, max_length=128)
    memory_strategy: MemoryStrategy | None = None
    window_size: int | None = Field(default=None, ge=4, le=100)


class ConversationMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: MemoryStrategy = "buffer"
    window_size: int = Field(default=10, ge=4, le=100)
    messages: list[ChatMessage] = Field(default_factory=list)
    summary: str = ""

    def add_message(self, role: MessageRole, content: str) -> ChatMessage:
        message = ChatMessage(role=role, content=content)
        self.messages.append(message)
        return message

    def context_messages(self) -> list[ChatMessage]:
        if self.strategy == "buffer":
            return list(self.messages)

        if self.strategy == "window":
            system_messages = [message for message in self.messages if message.role == "system"]
            conversational = [message for message in self.messages if message.role != "system"]
            return system_messages[-1:] + conversational[-self.window_size :]

        if self.strategy == "summary":
            system_messages = [message for message in self.messages if message.role == "system"]
            recent = [message for message in self.messages if message.role != "system"][-4:]
            context = system_messages[-1:]
            if self.summary.strip():
                context.append(
                    ChatMessage(
                        role="system",
                        content=f"Conversation summary so far: {self.summary.strip()}",
                    )
                )
            context.extend(recent)
            return context

        raise InvalidMessageError(f"Unsupported memory strategy: {self.strategy}")

    def stats(self, total_tokens_used: int) -> MemoryStats:
        return MemoryStats(
            strategy=self.strategy,
            stored_messages=len(self.messages),
            has_summary=bool(self.summary.strip()),
            context_window=self.window_size,
            total_tokens_used=total_tokens_used,
        )


class UserSession(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    user_id: str = Field(min_length=1, max_length=128)
    memory: ConversationMemory
    total_tokens_used: int = Field(default=0, ge=0)
    lock: Any = Field(default_factory=threading.RLock)


class SessionStore:
    def __init__(self, system_prompt: str, default_strategy: MemoryStrategy, default_window_size: int) -> None:
        self._system_prompt = system_prompt
        self._default_strategy = default_strategy
        self._default_window_size = default_window_size
        self._sessions: dict[str, UserSession] = {}
        self._lock = threading.RLock()

    def get_or_create(
        self,
        user_id: str,
        strategy: MemoryStrategy | None = None,
        window_size: int | None = None,
    ) -> UserSession:
        with self._lock:
            session = self._sessions.get(user_id)
            if session is None:
                session = self._new_session(
                    user_id=user_id,
                    strategy=strategy or self._default_strategy,
                    window_size=window_size or self._default_window_size,
                )
                self._sessions[user_id] = session
                return session

            with session.lock:
                if strategy is not None and strategy != session.memory.strategy:
                    session.memory.strategy = strategy
                if window_size is not None and window_size != session.memory.window_size:
                    session.memory.window_size = window_size
                return session

    def reset(
        self,
        user_id: str,
        strategy: MemoryStrategy | None = None,
        window_size: int | None = None,
    ) -> UserSession:
        with self._lock:
            session = self._new_session(
                user_id=user_id,
                strategy=strategy or self._default_strategy,
                window_size=window_size or self._default_window_size,
            )
            self._sessions[user_id] = session
            return session

    def delete(self, user_id: str) -> None:
        with self._lock:
            if user_id not in self._sessions:
                raise SessionNotFoundError(f"Unknown user_id: {user_id}")
            del self._sessions[user_id]

    def _new_session(self, user_id: str, strategy: MemoryStrategy, window_size: int) -> UserSession:
        memory = ConversationMemory(strategy=strategy, window_size=window_size)
        memory.add_message("system", self._system_prompt)
        return UserSession(user_id=user_id, memory=memory)


class ConversationalChatbot:
    SYSTEM_PROMPT = (
        "You are a helpful, intelligent AI assistant named Leeza-Bot. "
        "You have excellent memory and can reference earlier parts of the conversation. "
        "Be concise, friendly, and personalized. Use the user's name if they mention it. "
        "If asked about previous topics, recall them accurately from the supplied conversation history."
    )

    SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Summarize older conversation turns into a compact factual memory. "
                "Preserve names, preferences, user goals, decisions, and unresolved tasks. "
                "Do not invent details.",
            ),
            ("human", "Existing summary:\n{existing_summary}\n\nOlder turns:\n{older_turns}\n\nUpdated summary:"),
        ]
    )

    CHAT_PROMPT = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._llm_lock = threading.RLock()
        self._llm = ChatGroq(
            model=config.groq_model,
            temperature=config.groq_temperature,
            groq_api_key=config.groq_api_key.get_secret_value(),
            max_tokens=config.max_response_tokens,
        )
        self._embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
        self._summary_chain = self.SUMMARY_PROMPT | self._llm
        self._chat_chain = self.CHAT_PROMPT | self._llm
        self._sessions = SessionStore(
            system_prompt=self.SYSTEM_PROMPT,
            default_strategy=config.default_memory_strategy,
            default_window_size=config.default_window_size,
        )

    def chat(self, request: ChatRequest) -> ChatResponse:
        session = self._sessions.get_or_create(
            user_id=request.user_id,
            strategy=request.memory_strategy,
            window_size=request.window_size,
        )

        with session.lock:
            user_message = session.memory.add_message("user", request.message)

            if session.memory.strategy == "summary" and self._summary_needed(session.memory):
                self._compress_old_messages(session)

            langchain_messages = self._to_langchain_messages(session.memory.context_messages())

            with self._llm_lock:
                ai_response = self._chat_chain.invoke({"messages": langchain_messages})

            reply_text = self._extract_text(ai_response)
            reply_message = session.memory.add_message("assistant", reply_text)
            session.total_tokens_used += self._extract_token_usage(ai_response)

            return ChatResponse(
                user_id=session.user_id,
                message=user_message,
                reply=reply_message,
                memory_stats=session.memory.stats(session.total_tokens_used),
            )

    def chat_batch(self, request: BatchChatRequest) -> BatchChatResponse:
        from concurrent.futures import ThreadPoolExecutor

        max_workers = min(self.config.batch_workers, len(request.requests))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            responses = list(executor.map(self.chat, request.requests))
        return BatchChatResponse(responses=responses)

    def reset(self, request: ResetRequest) -> MemoryStats:
        session = self._sessions.reset(
            user_id=request.user_id,
            strategy=request.memory_strategy,
            window_size=request.window_size,
        )
        with session.lock:
            return session.memory.stats(session.total_tokens_used)

    def delete_session(self, user_id: str) -> None:
        self._sessions.delete(user_id)

    def _summary_needed(self, memory: ConversationMemory) -> bool:
        non_system_count = len([message for message in memory.messages if message.role != "system"])
        return non_system_count > memory.window_size

    def _compress_old_messages(self, session: UserSession) -> None:
        memory = session.memory
        non_system_messages = [message for message in memory.messages if message.role != "system"]
        old_messages = non_system_messages[:-4]

        if not old_messages:
            return

        older_turns = "\n".join(f"{message.role.upper()}: {message.content}" for message in old_messages)

        with self._llm_lock:
            summary_response = self._summary_chain.invoke(
                {
                    "existing_summary": memory.summary or "(none)",
                    "older_turns": older_turns,
                }
            )

        memory.summary = self._extract_text(summary_response)
        session.total_tokens_used += self._extract_token_usage(summary_response)

        system_messages = [message for message in memory.messages if message.role == "system"]
        recent_messages = non_system_messages[-4:]
        memory.messages = system_messages[-1:] + recent_messages

    @staticmethod
    def _to_langchain_messages(messages: Iterable[ChatMessage]) -> list[BaseMessage]:
        converted: list[BaseMessage] = []

        for message in messages:
            if message.role == "system":
                converted.append(SystemMessage(content=message.content))
            elif message.role == "user":
                converted.append(HumanMessage(content=message.content))
            elif message.role == "assistant":
                converted.append(AIMessage(content=message.content))
            else:
                raise InvalidMessageError(f"Unsupported message role: {message.role}")

        return converted

    @staticmethod
    def _extract_text(response: object) -> str:
        content = getattr(response, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        raise ChatbotError("LLM returned an empty or unsupported response.")

    @staticmethod
    def _extract_token_usage(response: object) -> int:
        response_metadata = getattr(response, "response_metadata", None)
        if not isinstance(response_metadata, dict):
            return 0

        token_usage = response_metadata.get("token_usage")
        if not isinstance(token_usage, dict):
            return 0

        total_tokens = token_usage.get("total_tokens")
        return total_tokens if isinstance(total_tokens, int) and total_tokens >= 0 else 0


STREAMLIT_CODE = '''
import streamlit as st

from conversational_chatbot import AppConfig, ChatRequest, ConversationalChatbot, ResetRequest

st.set_page_config(page_title="AI Chatbot", page_icon="*", layout="wide")
st.title("Conversational AI Chatbot with Memory")

if "chatbot" not in st.session_state:
    st.session_state.chatbot = ConversationalChatbot(AppConfig())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = "streamlit-user"

with st.sidebar:
    st.header("Settings")
    st.session_state.user_id = st.text_input("User ID", st.session_state.user_id)
    memory_strategy = st.selectbox("Memory Strategy", ["buffer", "window", "summary"])
    window_size = st.slider("Window Size", 4, 100, 10)

    if st.button("Reset Conversation", use_container_width=True):
        st.session_state.chatbot.reset(
            ResetRequest(
                user_id=st.session_state.user_id,
                memory_strategy=memory_strategy,
                window_size=window_size,
            )
        )
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        st.caption(message.get("timestamp", ""))

if prompt := st.chat_input("Type your message..."):
    request = ChatRequest(
        user_id=st.session_state.user_id,
        message=prompt,
        memory_strategy=memory_strategy,
        window_size=window_size,
    )

    response = st.session_state.chatbot.chat(request)

    st.session_state.messages.append(response.message.model_dump())
    st.session_state.messages.append(response.reply.model_dump())
    st.rerun()
'''


def build_chatbot() -> ConversationalChatbot:
    return ConversationalChatbot(AppConfig())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conversational AI Chatbot with ChatGroq memory.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    chat_parser = subparsers.add_parser("chat")
    chat_parser.add_argument("message")
    chat_parser.add_argument("--user-id", default="default")
    chat_parser.add_argument("--memory-strategy", choices=["buffer", "window", "summary"], default=None)
    chat_parser.add_argument("--window-size", type=int, default=None)

    batch_parser = subparsers.add_parser("batch")
    batch_parser.add_argument("batch_json", type=Path)

    reset_parser = subparsers.add_parser("reset")
    reset_parser.add_argument("--user-id", required=True)
    reset_parser.add_argument("--memory-strategy", choices=["buffer", "window", "summary"], default=None)
    reset_parser.add_argument("--window-size", type=int, default=None)

    streamlit_parser = subparsers.add_parser("write-streamlit-ui")
    streamlit_parser.add_argument("--path", type=Path, default=Path("conversational_chatbot_ui.py"))

    demo_parser = subparsers.add_parser("demo")
    demo_parser.add_argument("--user-id", default="demo-user")
    demo_parser.add_argument("--memory-strategy", choices=["buffer", "window", "summary"], default="buffer")

    return parser.parse_args()


def _request_from_mapping(payload: object) -> ChatRequest:
    if not isinstance(payload, dict):
        raise InvalidMessageError("Each batch item must be a JSON object.")
    return ChatRequest.model_validate(payload)


def run_demo(chatbot: ConversationalChatbot, user_id: str, memory_strategy: MemoryStrategy) -> None:
    demo_messages = [
        "Hi, my name is Leeza.",
        "Tell me about AI agents.",
        "What was the first thing I told you?",
        "How does your memory work?",
    ]

    for message in demo_messages:
        response = chatbot.chat(
            ChatRequest(
                user_id=user_id,
                message=message,
                memory_strategy=memory_strategy,
            )
        )
        print(response.model_dump_json(indent=2))


def main() -> None:
    args = parse_args()
    chatbot = build_chatbot()

    if args.command == "chat":
        response = chatbot.chat(
            ChatRequest(
                user_id=args.user_id,
                message=args.message,
                memory_strategy=args.memory_strategy,
                window_size=args.window_size,
            )
        )
        print(response.model_dump_json(indent=2))
        return

    if args.command == "batch":
        batch_path = args.batch_json.expanduser().resolve()
        with batch_path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)

        if not isinstance(payload, list):
            raise InvalidMessageError("Batch JSON must contain a list of chat request objects.")

        response = chatbot.chat_batch(
            BatchChatRequest(requests=[_request_from_mapping(item) for item in payload])
        )
        print(response.model_dump_json(indent=2))
        return

    if args.command == "reset":
        response = chatbot.reset(
            ResetRequest(
                user_id=args.user_id,
                memory_strategy=args.memory_strategy,
                window_size=args.window_size,
            )
        )
        print(response.model_dump_json(indent=2))
        return

    if args.command == "write-streamlit-ui":
        output_path = args.path.expanduser().resolve()
        output_path.write_text(STREAMLIT_CODE, encoding="utf-8")
        print(json.dumps({"status": "ok", "path": str(output_path)}, indent=2))
        return

    if args.command == "demo":
        run_demo(chatbot, user_id=args.user_id, memory_strategy=args.memory_strategy)
        return

    raise ChatbotError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()

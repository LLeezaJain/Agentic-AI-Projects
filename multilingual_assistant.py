from __future__ import annotations
import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

load_dotenv()


class MultilingualAssistantError(RuntimeError):
    pass


class ConfigurationError(MultilingualAssistantError):
    pass


class SessionNotFoundError(MultilingualAssistantError):
    pass


class EmptyMessageError(MultilingualAssistantError):
    pass


Confidence = Literal["high", "medium", "assumed"]
ScriptKind = Literal["Latin", "non-Latin"]


LANGUAGE_SIGNATURES: tuple[tuple[str, str, str, ScriptKind, Confidence], ...] = (
    ("Hindi", "hi", r"[\u0900-\u097F]", "non-Latin", "high"),
    ("Chinese", "zh", r"[\u4E00-\u9FFF]", "non-Latin", "high"),
    ("Japanese", "ja", r"[\u3040-\u30FF]", "non-Latin", "high"),
    ("Korean", "ko", r"[\uAC00-\uD7AF]", "non-Latin", "high"),
    ("Arabic", "ar", r"[\u0600-\u06FF]", "non-Latin", "high"),
    ("Russian", "ru", r"[\u0400-\u04FF]", "non-Latin", "high"),
    ("Greek", "el", r"[\u0370-\u03FF]", "non-Latin", "high"),
    ("Hebrew", "he", r"[\u0590-\u05FF]", "non-Latin", "high"),
    ("Thai", "th", r"[\u0E00-\u0E7F]", "non-Latin", "high"),
    ("French", "fr", r"\b(le|la|les|je|tu|il|nous|vous|bonjour|merci|oui|non)\b", "Latin", "medium"),
    ("Spanish", "es", r"\b(el|la|los|las|hola|gracias|si|sí|no|que|qué|como|cómo|espanol|español)\b", "Latin", "medium"),
    ("German", "de", r"\b(der|die|das|ich|du|wir|sie|guten|bitte|danke|nicht)\b", "Latin", "medium"),
    ("Portuguese", "pt", r"\b(o|a|os|as|eu|voce|você|obrigado|ola|olá|nao|não|sim)\b", "Latin", "medium"),
    ("Italian", "it", r"\b(il|la|lo|io|tu|noi|ciao|grazie|si|sì|no|cosa)\b", "Latin", "medium"),
)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    groq_api_key: SecretStr = Field(default_factory=lambda: SecretStr(os.getenv("GROQ_API_KEY", "")))
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    default_response_language: str = Field(default="English", min_length=1)
    max_history_messages: int = Field(default=12, ge=2, le=200)
    max_response_tokens: int = Field(default=500, ge=32, le=4096)
    batch_workers: int = Field(default=4, ge=1, le=32)

    @field_validator("groq_api_key")
    @classmethod
    def require_groq_api_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ConfigurationError("GROQ_API_KEY is required.")
        return value


class DetectedLanguage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    language: str = Field(min_length=1)
    code: str = Field(min_length=2, max_length=12)
    confidence: Confidence
    script: ScriptKind


class ConversationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)
    language: str = Field(min_length=1)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str = Field(default="default", min_length=1, max_length=128)
    message: str = Field(min_length=1)
    response_language: str | None = Field(default=None, min_length=1)


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str
    response: str
    detected_input_language: DetectedLanguage
    response_language: str
    turn: int = Field(ge=1)


class BatchChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requests: list[ChatRequest] = Field(min_length=1, max_length=100)


class BatchChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    responses: list[ChatResponse]


class TranslateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str = Field(default="default", min_length=1, max_length=128)
    text: str = Field(min_length=1)
    target_language: str = Field(min_length=1)
    source_language: str | None = Field(default=None, min_length=1)


class TranslateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str
    source_language: str
    target_language: str
    translation: str


class UserSession(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    user_id: str
    preferred_response_language: str
    history: list[ConversationMessage] = Field(default_factory=list)
    turn_count: int = Field(default=0, ge=0)
    lock: Any = Field(default_factory=threading.RLock)


class SessionStore:
    def __init__(self, default_response_language: str) -> None:
        self._default_response_language = default_response_language
        self._sessions: dict[str, UserSession] = {}
        self._lock = threading.RLock()

    def get_or_create(self, user_id: str, response_language: str | None = None) -> UserSession:
        with self._lock:
            session = self._sessions.get(user_id)
            if session is None:
                session = UserSession(
                    user_id=user_id,
                    preferred_response_language=response_language or self._default_response_language,
                )
                self._sessions[user_id] = session
            elif response_language is not None:
                with session.lock:
                    session.preferred_response_language = response_language
            return session

    def reset(self, user_id: str, response_language: str | None = None) -> UserSession:
        with self._lock:
            session = UserSession(
                user_id=user_id,
                preferred_response_language=response_language or self._default_response_language,
            )
            self._sessions[user_id] = session
            return session

    def delete(self, user_id: str) -> None:
        with self._lock:
            if user_id not in self._sessions:
                raise SessionNotFoundError(f"Unknown user_id: {user_id}")
            del self._sessions[user_id]


class LanguageDetector:
    @staticmethod
    def detect(text: str) -> DetectedLanguage:
        if not text.strip():
            raise EmptyMessageError("Cannot detect language for an empty message.")

        sample = text[:500]

        for language, code, pattern, script, confidence in LANGUAGE_SIGNATURES:
            if confidence == "high" and re.search(pattern, sample):
                return DetectedLanguage(language=language, code=code, confidence=confidence, script=script)

        for language, code, pattern, script, confidence in LANGUAGE_SIGNATURES:
            if confidence == "medium" and re.search(pattern, sample, re.IGNORECASE):
                return DetectedLanguage(language=language, code=code, confidence=confidence, script=script)

        return DetectedLanguage(language="English", code="en", confidence="assumed", script="Latin")


class TranslationEngine:
    TRANSLATE_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a precise professional translator. Return only the translated text. "
                "Preserve meaning, tone, formatting, numbers, and named entities.",
            ),
            (
                "human",
                "Translate from {source_language} to {target_language}:\n\n{text}",
            ),
        ]
    )

    def __init__(self, llm: ChatGroq) -> None:
        self._chain = self.TRANSLATE_PROMPT | llm
        self._cache: dict[str, str] = {}
        self._lock = threading.RLock()

    def translate(self, request: TranslateRequest) -> TranslateResponse:
        detected = LanguageDetector.detect(request.text)
        source_language = request.source_language or detected.language

        if source_language == request.target_language:
            return TranslateResponse(
                user_id=request.user_id,
                source_language=source_language,
                target_language=request.target_language,
                translation=request.text,
            )

        cache_key = json.dumps(
            {
                "source": source_language,
                "target": request.target_language,
                "text": request.text,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return TranslateResponse(
                    user_id=request.user_id,
                    source_language=source_language,
                    target_language=request.target_language,
                    translation=cached,
                )

        response = self._chain.invoke(
            {
                "source_language": source_language,
                "target_language": request.target_language,
                "text": request.text,
            }
        )
        translation = self._extract_text(response)

        with self._lock:
            self._cache[cache_key] = translation

        return TranslateResponse(
            user_id=request.user_id,
            source_language=source_language,
            target_language=request.target_language,
            translation=translation,
        )

    @staticmethod
    def _extract_text(response: object) -> str:
        content = getattr(response, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        raise MultilingualAssistantError("LLM returned an empty or unsupported translation response.")


class MultilingualPromptBuilder:
    SYSTEM_TEMPLATE = (
        "You are an expert multilingual AI assistant with native-level fluency. "
        "Respond only in {response_language}. "
        "The user's detected input language is {input_language}. "
        "Adapt tone naturally and culturally for {response_language} speakers. "
        "If the user's request is ambiguous, ask a concise clarification in {response_language}."
    )

    @classmethod
    def build_messages(
        cls,
        user_message: str,
        detected_language: DetectedLanguage,
        response_language: str,
        history: list[ConversationMessage],
        max_history_messages: int,
    ) -> list[BaseMessage]:
        messages: list[BaseMessage] = [
            SystemMessage(
                content=cls.SYSTEM_TEMPLATE.format(
                    response_language=response_language,
                    input_language=detected_language.language,
                )
            )
        ]

        for message in history[-max_history_messages:]:
            if message.role == "user":
                messages.append(HumanMessage(content=f"[{message.language}] {message.content}"))
            else:
                messages.append(AIMessage(content=f"[{message.language}] {message.content}"))

        messages.append(HumanMessage(content=user_message))
        return messages


class MultilingualAssistant:
    CHAT_PROMPT = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name="messages")])

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self._llm_lock = threading.RLock()
        self._llm = ChatGroq(
            model=self.config.groq_model,
            temperature=self.config.groq_temperature,
            groq_api_key=self.config.groq_api_key.get_secret_value(),
            max_tokens=self.config.max_response_tokens,
        )
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
        self._chat_chain = self.CHAT_PROMPT | self._llm
        self._translator = TranslationEngine(self._llm)
        self._sessions = SessionStore(default_response_language=self.config.default_response_language)

    def chat(self, request: ChatRequest) -> ChatResponse:
        detected = LanguageDetector.detect(request.message)
        session = self._sessions.get_or_create(request.user_id, request.response_language)

        with session.lock:
            response_language = request.response_language or session.preferred_response_language
            messages = MultilingualPromptBuilder.build_messages(
                user_message=request.message,
                detected_language=detected,
                response_language=response_language,
                history=session.history,
                max_history_messages=self.config.max_history_messages,
            )

            with self._llm_lock:
                response = self._chat_chain.invoke({"messages": messages})

            response_text = self._extract_text(response)
            session.turn_count += 1
            session.history.append(
                ConversationMessage(
                    role="user",
                    content=request.message,
                    language=detected.language,
                )
            )
            session.history.append(
                ConversationMessage(
                    role="assistant",
                    content=response_text,
                    language=response_language,
                )
            )

            if len(session.history) > self.config.max_history_messages * 2:
                session.history = session.history[-self.config.max_history_messages * 2 :]

            return ChatResponse(
                user_id=request.user_id,
                response=response_text,
                detected_input_language=detected,
                response_language=response_language,
                turn=session.turn_count,
            )

    def chat_batch(self, request: BatchChatRequest) -> BatchChatResponse:
        max_workers = min(self.config.batch_workers, len(request.requests))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            responses = list(executor.map(self.chat, request.requests))
        return BatchChatResponse(responses=responses)

    def translate(self, request: TranslateRequest) -> TranslateResponse:
        self._sessions.get_or_create(request.user_id)
        return self._translator.translate(request)

    def reset_user(self, user_id: str, response_language: str | None = None) -> None:
        self._sessions.reset(user_id, response_language)

    def delete_user(self, user_id: str) -> None:
        self._sessions.delete(user_id)

    @staticmethod
    def detect_language(text: str) -> DetectedLanguage:
        return LanguageDetector.detect(text)

    @staticmethod
    def _extract_text(response: object) -> str:
        content = getattr(response, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        raise MultilingualAssistantError("LLM returned an empty or unsupported chat response.")


def detect_language(text: str) -> DetectedLanguage:
    return LanguageDetector.detect(text)


def build_assistant() -> MultilingualAssistant:
    return MultilingualAssistant(AppConfig())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multilingual AI Assistant using ChatGroq.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect_parser = subparsers.add_parser("detect")
    detect_parser.add_argument("text")

    chat_parser = subparsers.add_parser("chat")
    chat_parser.add_argument("message")
    chat_parser.add_argument("--user-id", default="default")
    chat_parser.add_argument("--response-language", default=None)

    translate_parser = subparsers.add_parser("translate")
    translate_parser.add_argument("text")
    translate_parser.add_argument("--user-id", default="default")
    translate_parser.add_argument("--target-language", required=True)
    translate_parser.add_argument("--source-language", default=None)

    batch_parser = subparsers.add_parser("batch")
    batch_parser.add_argument("batch_json", type=Path)

    reset_parser = subparsers.add_parser("reset")
    reset_parser.add_argument("--user-id", required=True)
    reset_parser.add_argument("--response-language", default=None)

    return parser.parse_args()


def _chat_request_from_mapping(payload: object) -> ChatRequest:
    if not isinstance(payload, dict):
        raise MultilingualAssistantError("Each batch item must be a JSON object.")
    return ChatRequest.model_validate(payload)


def main() -> None:
    args = parse_args()
    assistant = build_assistant()

    if args.command == "detect":
        response = assistant.detect_language(args.text)
        print(response.model_dump_json(indent=2))
        return

    if args.command == "chat":
        response = assistant.chat(
            ChatRequest(
                user_id=args.user_id,
                message=args.message,
                response_language=args.response_language,
            )
        )
        print(response.model_dump_json(indent=2))
        return

    if args.command == "translate":
        response = assistant.translate(
            TranslateRequest(
                user_id=args.user_id,
                text=args.text,
                source_language=args.source_language,
                target_language=args.target_language,
            )
        )
        print(response.model_dump_json(indent=2))
        return

    if args.command == "batch":
        batch_path = args.batch_json.expanduser().resolve()
        with batch_path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)

        if not isinstance(payload, list):
            raise MultilingualAssistantError("Batch JSON must contain a list of chat request objects.")

        response = assistant.chat_batch(
            BatchChatRequest(requests=[_chat_request_from_mapping(item) for item in payload])
        )
        print(response.model_dump_json(indent=2))
        return

    if args.command == "reset":
        assistant.reset_user(args.user_id, args.response_language)
        print(json.dumps({"status": "ok", "user_id": args.user_id}, indent=2))
        return

    raise MultilingualAssistantError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
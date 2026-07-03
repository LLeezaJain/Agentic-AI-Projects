from __future__ import annotations
import argparse
import io
import json
import os
import struct
import tempfile
import threading
import time
import wave
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


class VoiceAssistantError(RuntimeError):
    pass


class ConfigurationError(VoiceAssistantError):
    pass


class AudioCaptureError(VoiceAssistantError):
    pass


class SpeechToTextError(VoiceAssistantError):
    pass


class TextToSpeechError(VoiceAssistantError):
    pass


class SessionNotFoundError(VoiceAssistantError):
    pass


AudioFormat = Literal["wav", "mp3", "m4a", "ogg", "flac", "webm"]
MessageRole = Literal["user", "assistant"]


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    groq_api_key: SecretStr = Field(default_factory=lambda: SecretStr(os.getenv("GROQ_API_KEY", "")))
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    groq_stt_model: str = "whisper-large-v3-turbo"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_response_tokens: int = Field(default=180, ge=32, le=1024)
    max_history_messages: int = Field(default=10, ge=2, le=100)
    batch_workers: int = Field(default=4, ge=1, le=32)
    default_voice: str = "default"
    enable_tts: bool = False

    @field_validator("groq_api_key")
    @classmethod
    def require_groq_api_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ConfigurationError("GROQ_API_KEY is required.")
        return value


class TTSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    voice: str = Field(default="default", min_length=1)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    output_format: Literal["wav"] = "wav"


class RecorderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    channels: int = Field(default=1, ge=1, le=2)
    chunk_size: int = Field(default=1024, ge=256, le=8192)
    silence_threshold: float = Field(default=500.0, ge=1.0)
    silence_duration: float = Field(default=2.0, ge=0.25, le=10.0)
    max_seconds: int = Field(default=30, ge=1, le=300)


class TranscriptionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    audio_path: Path | None = None
    audio_bytes: bytes | None = None
    file_format: AudioFormat = "wav"

    @field_validator("audio_path")
    @classmethod
    def validate_audio_path(cls, value: Path | None) -> Path | None:
        if value is None:
            return None

        resolved = value.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Audio file not found: {resolved}")
        if not resolved.is_file():
            raise SpeechToTextError(f"Audio path is not a file: {resolved}")
        if resolved.suffix.lower().lstrip(".") not in {"wav", "mp3", "m4a", "ogg", "flac", "webm"}:
            raise SpeechToTextError(f"Unsupported audio format: {resolved.suffix}")
        return resolved


class TranscriptionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    language: str | None = None
    duration: float | None = Field(default=None, ge=0.0)
    model: str


class SynthesisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    output_path: Path | None = None
    config: TTSConfig = Field(default_factory=TTSConfig)

    @field_validator("output_path")
    @classmethod
    def normalize_output_path(cls, value: Path | None) -> Path | None:
        return value.expanduser().resolve() if value is not None else None


class SynthesisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    audio_bytes: bytes
    output_path: str | None = None
    format: Literal["wav"] = "wav"


class ConversationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: MessageRole
    content: str = Field(min_length=1)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class VoiceTurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str = Field(default="default", min_length=1, max_length=128)
    text: str | None = Field(default=None, min_length=1)
    audio_path: Path | None = None
    synthesize_audio: bool = False

    @field_validator("audio_path")
    @classmethod
    def validate_audio_path(cls, value: Path | None) -> Path | None:
        if value is None:
            return None

        resolved = value.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Audio file not found: {resolved}")
        if not resolved.is_file():
            raise SpeechToTextError(f"Audio path is not a file: {resolved}")
        return resolved


class VoiceTurnResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str
    user_input: str
    response: str
    transcription: TranscriptionResponse | None = None
    audio_output_path: str | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class BatchVoiceTurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requests: list[VoiceTurnRequest] = Field(min_length=1, max_length=100)


class BatchVoiceTurnResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    responses: list[VoiceTurnResponse]


class UserSession(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    user_id: str
    history: list[ConversationMessage] = Field(default_factory=list)
    session_log: list[VoiceTurnResponse] = Field(default_factory=list)
    lock: Any = Field(default_factory=threading.RLock)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, UserSession] = {}
        self._lock = threading.RLock()

    def get_or_create(self, user_id: str) -> UserSession:
        with self._lock:
            session = self._sessions.get(user_id)
            if session is None:
                session = UserSession(user_id=user_id)
                self._sessions[user_id] = session
            return session

    def reset(self, user_id: str) -> UserSession:
        with self._lock:
            session = UserSession(user_id=user_id)
            self._sessions[user_id] = session
            return session

    def delete(self, user_id: str) -> None:
        with self._lock:
            if user_id not in self._sessions:
                raise SessionNotFoundError(f"Unknown user_id: {user_id}")
            del self._sessions[user_id]


class SpeechToText:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._lock = threading.RLock()

        try:
            from groq import Groq
        except ImportError as exc:
            raise ConfigurationError("Install groq to use speech-to-text: pip install groq") from exc

        self._client = Groq(api_key=config.groq_api_key.get_secret_value())

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResponse:
        if request.audio_path is None and request.audio_bytes is None:
            raise SpeechToTextError("Provide audio_path or audio_bytes.")
        if request.audio_path is not None and request.audio_bytes is not None:
            raise SpeechToTextError("Provide only one of audio_path or audio_bytes.")

        with self._lock:
            if request.audio_path is not None:
                with request.audio_path.open("rb") as audio_file:
                    response = self._client.audio.transcriptions.create(
                        model=self._config.groq_stt_model,
                        file=audio_file,
                        response_format="verbose_json",
                    )
            else:
                audio_file = io.BytesIO(request.audio_bytes or b"")
                audio_file.name = f"recording.{request.file_format}"
                response = self._client.audio.transcriptions.create(
                    model=self._config.groq_stt_model,
                    file=audio_file,
                    response_format="verbose_json",
                )

        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise SpeechToTextError("Speech-to-text returned no transcription text.")

        duration = getattr(response, "duration", None)
        language = getattr(response, "language", None)

        return TranscriptionResponse(
            text=text.strip(),
            language=language if isinstance(language, str) else None,
            duration=float(duration) if isinstance(duration, int | float) else None,
            model=self._config.groq_stt_model,
        )


class TextToSpeech:
    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled
        self._lock = threading.RLock()

    def synthesize(self, request: SynthesisRequest) -> SynthesisResponse:
        if not self._enabled:
            raise TextToSpeechError("Text-to-speech is disabled. Set enable_tts=True in AppConfig.")

        with self._lock:
            try:
                audio_bytes = self._synthesize_with_pyttsx3(request.text, request.config)
            except ImportError as exc:
                raise ConfigurationError("Install pyttsx3 to use local text-to-speech: pip install pyttsx3") from exc

        output_path_value: str | None = None
        if request.output_path is not None:
            request.output_path.parent.mkdir(parents=True, exist_ok=True)
            request.output_path.write_bytes(audio_bytes)
            output_path_value = str(request.output_path)

        return SynthesisResponse(
            audio_bytes=audio_bytes,
            output_path=output_path_value,
            format=request.config.output_format,
        )

    @staticmethod
    def _synthesize_with_pyttsx3(text: str, config: TTSConfig) -> bytes:
        import pyttsx3

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", int(180 * config.speed))

            voices = engine.getProperty("voices") or []
            if config.voice != "default":
                for voice in voices:
                    if config.voice.lower() in str(getattr(voice, "name", "")).lower():
                        engine.setProperty("voice", voice.id)
                        break

            engine.save_to_file(text, str(tmp_path))
            engine.runAndWait()
            audio = tmp_path.read_bytes()
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        if not audio:
            raise TextToSpeechError("Local text-to-speech produced empty audio.")
        return audio

    @staticmethod
    def play_audio(audio_bytes: bytes) -> None:
        if not audio_bytes:
            raise TextToSpeechError("Cannot play empty audio bytes.")

        try:
            import winsound
        except ImportError as exc:
            raise TextToSpeechError("Audio playback is only implemented with winsound on Windows.") from exc

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            winsound.PlaySound(tmp_path, winsound.SND_FILENAME)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class MicrophoneRecorder:
    def __init__(self, config: RecorderConfig | None = None) -> None:
        self.config = config or RecorderConfig()
        self._lock = threading.RLock()

    def record(self) -> bytes:
        try:
            import pyaudio
        except ImportError as exc:
            raise ConfigurationError("Install pyaudio to record microphone audio: pip install pyaudio") from exc

        with self._lock:
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size,
            )

            frames: list[bytes] = []
            silent_chunks = 0
            silent_chunks_needed = int(
                self.config.silence_duration * self.config.sample_rate / self.config.chunk_size
            )
            max_chunks = int(self.config.max_seconds * self.config.sample_rate / self.config.chunk_size)

            try:
                for _ in range(max_chunks):
                    data = stream.read(self.config.chunk_size, exception_on_overflow=False)
                    frames.append(data)

                    samples = struct.unpack("<" + "h" * (len(data) // 2), data)
                    amplitude = max(abs(sample) for sample in samples) if samples else 0

                    if amplitude < self.config.silence_threshold:
                        silent_chunks += 1
                        if silent_chunks >= silent_chunks_needed and len(frames) > 10:
                            break
                    else:
                        silent_chunks = 0
            finally:
                stream.stop_stream()
                stream.close()
                sample_width = pa.get_sample_size(pyaudio.paInt16)
                pa.terminate()

        if not frames:
            raise AudioCaptureError("No microphone audio was captured.")

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(self.config.channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(self.config.sample_rate)
            wav_file.writeframes(b"".join(frames))

        audio = buffer.getvalue()
        if not audio:
            raise AudioCaptureError("Microphone recording produced empty audio.")
        return audio


class VoiceConversationBrain:
    SYSTEM_PROMPT = (
        "You are a friendly voice assistant named Aria. "
        "Keep responses concise and conversational because they will be spoken aloud. "
        "Avoid markdown, bullet points, and long lists. "
        "Use natural spoken language. "
        "Answer in 2 to 3 sentences unless the user explicitly asks for more detail."
    )

    CHAT_PROMPT = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name="messages")])

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._llm_lock = threading.RLock()
        self._llm = ChatGroq(
            model=config.groq_model,
            temperature=config.groq_temperature,
            groq_api_key=config.groq_api_key.get_secret_value(),
            max_tokens=config.max_response_tokens,
        )
        self._chain = self.CHAT_PROMPT | self._llm

    def respond(self, session: UserSession, text: str) -> str:
        messages = self._build_messages(session, text)

        with self._llm_lock:
            response = self._chain.invoke({"messages": messages})

        reply = self._extract_text(response)
        session.history.append(ConversationMessage(role="user", content=text))
        session.history.append(ConversationMessage(role="assistant", content=reply))

        if len(session.history) > self._config.max_history_messages:
            session.history = session.history[-self._config.max_history_messages :]

        return reply

    def _build_messages(self, session: UserSession, text: str) -> list[BaseMessage]:
        messages: list[BaseMessage] = [SystemMessage(content=self.SYSTEM_PROMPT)]

        for message in session.history[-self._config.max_history_messages :]:
            if message.role == "user":
                messages.append(HumanMessage(content=message.content))
            elif message.role == "assistant":
                messages.append(AIMessage(content=message.content))

        messages.append(HumanMessage(content=text))
        return messages

    @staticmethod
    def _extract_text(response: object) -> str:
        content = getattr(response, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        raise VoiceAssistantError("LLM returned an empty or unsupported response.")


class VoiceAssistant:
    def __init__(self, config: AppConfig | None = None, use_mic: bool = False) -> None:
        self.config = config or AppConfig()
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
        self._stt = SpeechToText(self.config)
        self._tts = TextToSpeech(enabled=self.config.enable_tts)
        self._brain = VoiceConversationBrain(self.config)
        self._recorder = MicrophoneRecorder() if use_mic else None
        self._sessions = SessionStore()

    def transcribe_file(self, audio_path: Path) -> TranscriptionResponse:
        return self._stt.transcribe(TranscriptionRequest(audio_path=audio_path))

    def listen(self) -> TranscriptionResponse:
        if self._recorder is None:
            raise AudioCaptureError("Microphone recorder is disabled for this assistant instance.")
        audio_bytes = self._recorder.record()
        return self._stt.transcribe(TranscriptionRequest(audio_bytes=audio_bytes, file_format="wav"))

    def process_turn(self, request: VoiceTurnRequest) -> VoiceTurnResponse:
        session = self._sessions.get_or_create(request.user_id)

        with session.lock:
            transcription: TranscriptionResponse | None = None

            if request.text is not None:
                user_text = request.text
            elif request.audio_path is not None:
                transcription = self._stt.transcribe(TranscriptionRequest(audio_path=request.audio_path))
                user_text = transcription.text
            elif self._recorder is not None:
                transcription = self.listen()
                user_text = transcription.text
            else:
                raise VoiceAssistantError("Provide text, audio_path, or enable microphone recording.")

            response_text = self._brain.respond(session, user_text)
            audio_output_path: str | None = None

            if request.synthesize_audio:
                output_path = Path(f"aria_response_{request.user_id}_{int(time.time())}.wav").resolve()
                synthesis = self._tts.synthesize(
                    SynthesisRequest(
                        text=response_text,
                        output_path=output_path,
                        config=TTSConfig(voice=self.config.default_voice),
                    )
                )
                audio_output_path = synthesis.output_path

            response = VoiceTurnResponse(
                user_id=request.user_id,
                user_input=user_text,
                response=response_text,
                transcription=transcription,
                audio_output_path=audio_output_path,
            )
            session.session_log.append(response)
            return response

    def process_batch(self, request: BatchVoiceTurnRequest) -> BatchVoiceTurnResponse:
        max_workers = min(self.config.batch_workers, len(request.requests))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            responses = list(executor.map(self.process_turn, request.requests))
        return BatchVoiceTurnResponse(responses=responses)

    def run_conversation_loop(self, user_id: str = "default", turns: int = 5) -> list[VoiceTurnResponse]:
        if self._recorder is None:
            raise AudioCaptureError("Microphone recorder is disabled for this assistant instance.")

        responses: list[VoiceTurnResponse] = []
        for _ in range(turns):
            response = self.process_turn(VoiceTurnRequest(user_id=user_id))
            responses.append(response)

            if response.user_input.lower().strip() in {"goodbye", "bye", "exit", "quit"}:
                break

            time.sleep(0.5)

        return responses

    def reset_user(self, user_id: str) -> None:
        self._sessions.reset(user_id)

    def delete_user(self, user_id: str) -> None:
        self._sessions.delete(user_id)


def build_assistant(use_mic: bool = False, enable_tts: bool = False) -> VoiceAssistant:
    config = AppConfig(enable_tts=enable_tts)
    return VoiceAssistant(config=config, use_mic=use_mic)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voice AI Assistant using ChatGroq.")
    parser.add_argument("--mic", action="store_true", help="Enable microphone recording.")
    parser.add_argument("--tts", action="store_true", help="Enable local pyttsx3 text-to-speech.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    text_parser = subparsers.add_parser("text")
    text_parser.add_argument("text")
    text_parser.add_argument("--user-id", default="default")
    text_parser.add_argument("--synthesize-audio", action="store_true")

    audio_parser = subparsers.add_parser("audio")
    audio_parser.add_argument("audio_path", type=Path)
    audio_parser.add_argument("--user-id", default="default")
    audio_parser.add_argument("--synthesize-audio", action="store_true")

    listen_parser = subparsers.add_parser("listen")
    listen_parser.add_argument("--user-id", default="default")
    listen_parser.add_argument("--turns", type=int, default=1)
    listen_parser.add_argument("--synthesize-audio", action="store_true")

    batch_parser = subparsers.add_parser("batch")
    batch_parser.add_argument("batch_json", type=Path)

    reset_parser = subparsers.add_parser("reset")
    reset_parser.add_argument("--user-id", required=True)

    return parser.parse_args()


def _voice_request_from_mapping(payload: object) -> VoiceTurnRequest:
    if not isinstance(payload, dict):
        raise VoiceAssistantError("Each batch item must be a JSON object.")

    normalized = dict(payload)
    if normalized.get("audio_path") is not None:
        normalized["audio_path"] = Path(normalized["audio_path"])

    return VoiceTurnRequest.model_validate(normalized)


def main() -> None:
    args = parse_args()
    assistant = build_assistant(use_mic=args.mic or args.command == "listen", enable_tts=args.tts)

    if args.command == "text":
        response = assistant.process_turn(
            VoiceTurnRequest(
                user_id=args.user_id,
                text=args.text,
                synthesize_audio=args.synthesize_audio,
            )
        )
        print(response.model_dump_json(indent=2))
        return

    if args.command == "audio":
        response = assistant.process_turn(
            VoiceTurnRequest(
                user_id=args.user_id,
                audio_path=args.audio_path,
                synthesize_audio=args.synthesize_audio,
            )
        )
        print(response.model_dump_json(indent=2))
        return

    if args.command == "listen":
        if args.turns == 1:
            response = assistant.process_turn(
                VoiceTurnRequest(user_id=args.user_id, synthesize_audio=args.synthesize_audio)
            )
            print(response.model_dump_json(indent=2))
        else:
            responses = assistant.run_conversation_loop(user_id=args.user_id, turns=args.turns)
            print(json.dumps([response.model_dump() for response in responses], indent=2))
        return

    if args.command == "batch":
        batch_path = args.batch_json.expanduser().resolve()
        with batch_path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)

        if not isinstance(payload, list):
            raise VoiceAssistantError("Batch JSON must contain a list of request objects.")

        response = assistant.process_batch(
            BatchVoiceTurnRequest(requests=[_voice_request_from_mapping(item) for item in payload])
        )
        print(response.model_dump_json(indent=2))
        return

    if args.command == "reset":
        assistant.reset_user(args.user_id)
        print(json.dumps({"status": "ok", "user_id": args.user_id}, indent=2))
        return

    raise VoiceAssistantError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()

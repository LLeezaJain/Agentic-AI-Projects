from __future__ import annotations
import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

load_dotenv()


class ResearchSummarizerError(RuntimeError):
    pass


class ConfigurationError(ResearchSummarizerError):
    pass


class PdfExtractionError(ResearchSummarizerError):
    pass


class EmptyDocumentError(ResearchSummarizerError):
    pass


class UnsupportedInputError(ResearchSummarizerError):
    pass


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    groq_api_key: SecretStr = Field(default_factory=lambda: SecretStr(os.getenv("GROQ_API_KEY", "")))
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_tokens_per_chunk: int = Field(default=3000, ge=500, le=12000)
    max_sections: int = Field(default=12, ge=1, le=50)
    batch_workers: int = Field(default=4, ge=1, le=16)

    @field_validator("groq_api_key")
    @classmethod
    def require_groq_api_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ConfigurationError("GROQ_API_KEY is required.")
        return value


class PaperSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    page_start: int = Field(ge=1)
    word_count: int = Field(ge=1)


class SummaryChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    section: str = Field(min_length=1)
    content: str = Field(min_length=1)
    estimated_tokens: float = Field(gt=0)


class SectionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    section: str = Field(min_length=1)
    summary: str = Field(min_length=1)


class PaperSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=1)
    abstract_summary: str = Field(min_length=1)
    key_contributions: list[str] = Field(min_length=1)
    methodology: str = Field(min_length=1)
    results: str = Field(min_length=1)
    limitations: str = Field(min_length=1)
    one_liner: str = Field(min_length=1)
    full_summary: str = Field(min_length=1)


class SummarizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str = Field(default="default", min_length=1, max_length=128)
    pdf_path: Path | None = None
    use_sample: bool = False

    @field_validator("pdf_path")
    @classmethod
    def validate_pdf_path(cls, value: Path | None) -> Path | None:
        if value is None:
            return None

        resolved = value.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"PDF not found: {resolved}")
        if not resolved.is_file():
            raise UnsupportedInputError(f"Path is not a file: {resolved}")
        if resolved.suffix.lower() != ".pdf":
            raise UnsupportedInputError(f"Only PDF input is supported: {resolved}")
        return resolved


class SummarizeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str
    source: str
    sections: list[PaperSection]
    chunks: list[SummaryChunk]
    section_summaries: list[SectionSummary]
    summary: PaperSummary


class BatchSummarizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requests: list[SummarizeRequest] = Field(min_length=1, max_length=100)


class BatchSummarizeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    responses: list[SummarizeResponse]


class UserRuntimeState(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    user_id: str
    lock: threading.RLock = Field(default_factory=threading.RLock)
    completed_jobs: int = 0


class UserStateStore:
    def __init__(self) -> None:
        self._global_lock = threading.RLock()
        self._users: dict[str, UserRuntimeState] = {}

    def get_or_create(self, user_id: str) -> UserRuntimeState:
        with self._global_lock:
            state = self._users.get(user_id)
            if state is None:
                state = UserRuntimeState(user_id=user_id)
                self._users[user_id] = state
            return state

    def increment_completed_jobs(self, user_id: str) -> None:
        state = self.get_or_create(user_id)
        with state.lock:
            state.completed_jobs += 1


class PDFExtractor:
    SECTION_HEADERS = (
        "abstract",
        "introduction",
        "background",
        "related work",
        "methodology",
        "methods",
        "approach",
        "model",
        "experiments",
        "evaluation",
        "results",
        "discussion",
        "limitations",
        "conclusion",
        "references",
        "acknowledgments",
        "acknowledgements",
    )

    _page_pattern = re.compile(r"^\[PAGE\s+(\d+)\]\s*$", re.IGNORECASE)

    def extract_text(self, request: SummarizeRequest) -> tuple[str, str]:
        if request.pdf_path is not None:
            return self._extract_pdf_text(request.pdf_path), str(request.pdf_path)

        if request.use_sample:
            return self.sample_paper(), "sample:attention-is-all-you-need"

        raise UnsupportedInputError("Provide pdf_path or set use_sample=True explicitly.")

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        try:
            import pypdf
        except ImportError as exc:
            raise PdfExtractionError("Install pypdf to extract PDF text: pip install pypdf") from exc

        pages: list[str] = []
        with pdf_path.open("rb") as file_handle:
            reader = pypdf.PdfReader(file_handle)
            if len(reader.pages) == 0:
                raise EmptyDocumentError(f"PDF has no pages: {pdf_path}")

            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    pages.append(f"[PAGE {page_number}]\n{page_text}")

        text = "\n\n".join(pages).strip()
        if not text:
            raise EmptyDocumentError(f"No extractable text found in PDF: {pdf_path}")
        return text

    def extract_sections(self, text: str) -> list[PaperSection]:
        cleaned_text = text.strip()
        if not cleaned_text:
            raise EmptyDocumentError("Cannot section an empty document.")

        sections: list[PaperSection] = []
        current_title = "Introduction"
        current_page = 1
        section_start_page = 1
        current_content: list[str] = []

        for raw_line in cleaned_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            page_match = self._page_pattern.match(line)
            if page_match:
                current_page = int(page_match.group(1))
                continue

            if self._is_section_header(line) and current_content:
                section = self._build_section(current_title, current_content, section_start_page)
                if section is not None:
                    sections.append(section)
                current_title = self._normalize_title(line)
                section_start_page = current_page
                current_content = []
                continue

            if self._is_section_header(line) and not current_content:
                current_title = self._normalize_title(line)
                section_start_page = current_page
                continue

            current_content.append(line)

        final_section = self._build_section(current_title, current_content, section_start_page)
        if final_section is not None:
            sections.append(final_section)

        if not sections:
            raise EmptyDocumentError("No usable paper sections were found.")

        return sections

    def _is_section_header(self, line: str) -> bool:
        normalized = re.sub(r"^\d+(\.\d+)*\s+", "", line.lower()).strip(" .:-")
        return len(normalized) <= 64 and any(normalized == header or normalized.startswith(f"{header} ") for header in self.SECTION_HEADERS)

    @staticmethod
    def _normalize_title(line: str) -> str:
        title = re.sub(r"^\d+(\.\d+)*\s+", "", line).strip(" .:-")
        return title[:1].upper() + title[1:] if title else "Untitled"

    @staticmethod
    def _build_section(title: str, content: list[str], page_start: int) -> PaperSection | None:
        body = "\n".join(content).strip()
        word_count = len(body.split())
        if word_count == 0:
            return None
        return PaperSection(title=title, content=body, page_start=page_start, word_count=word_count)

    @staticmethod
    def sample_paper() -> str:
        return """[PAGE 1]
Attention Is All You Need

Abstract
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks
that include an encoder and a decoder. The best performing models also connect the encoder and decoder
through an attention mechanism. We propose a new simple network architecture, the Transformer, based
solely on attention mechanisms, dispensing with recurrence and convolutions entirely.

Introduction
Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular,
have been firmly established as state of the art approaches in sequence modeling and transduction problems
such as language modeling and machine translation.

[PAGE 2]
Methodology
The Transformer follows an encoder-decoder structure using stacked self-attention and point-wise,
fully connected layers for both the encoder and decoder. Multi-head attention allows the model to jointly
attend to information from different representation subspaces at different positions.

Results
On the WMT 2014 English-to-German translation task, the big transformer model outperforms the best
previously reported models including ensembles by more than 2.0 BLEU, establishing a new state-of-the-art
BLEU score of 28.4.

Limitations
Self-attention has quadratic complexity in the sequence length, which can make very long-context processing
computationally expensive.

Conclusion
In this work, we presented the Transformer, the first sequence transduction model based entirely on
attention, replacing recurrent layers most commonly used in encoder-decoder architectures with multi-headed
self-attention. The Transformer can be trained significantly faster than recurrent or convolutional models.
"""


class SmartChunker:
    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens

    def chunk_for_summary(self, sections: Iterable[PaperSection]) -> list[SummaryChunk]:
        chunks: list[SummaryChunk] = []

        for section in sections:
            words = section.content.split()
            estimated_tokens = self.estimate_tokens(words)

            if estimated_tokens <= self.max_tokens:
                chunks.append(
                    SummaryChunk(
                        section=section.title,
                        content=section.content,
                        estimated_tokens=estimated_tokens,
                    )
                )
                continue

            chunk_size = max(1, int(self.max_tokens / 1.33))
            overlap = max(1, chunk_size // 5)
            step = max(1, chunk_size - overlap)

            for start in range(0, len(words), step):
                part_words = words[start : start + chunk_size]
                if not part_words:
                    continue

                part_number = (start // step) + 1
                sub_content = " ".join(part_words)
                chunks.append(
                    SummaryChunk(
                        section=f"{section.title} (part {part_number})",
                        content=sub_content,
                        estimated_tokens=self.estimate_tokens(part_words),
                    )
                )

        if not chunks:
            raise EmptyDocumentError("No chunks were generated from the paper sections.")

        return chunks

    @staticmethod
    def estimate_tokens(words: list[str]) -> float:
        return max(1.0, len(words) * 1.33)


class ResearchSummarizer:
    SECTION_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert academic researcher. Produce accurate, concise section summaries. "
                "Do not invent details not supported by the section text.",
            ),
            (
                "human",
                "Section: {section}\n\nContent:\n{content}\n\n"
                "Return a concise 2-3 sentence summary of the section.",
            ),
        ]
    )

    FINAL_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert academic researcher. Create structured summaries from section summaries. "
                "Use only the supplied evidence. Return data that satisfies the requested schema exactly.",
            ),
            (
                "human",
                "Paper title candidate: {title}\n\nSection summaries:\n{section_summaries}\n\n"
                "Create a comprehensive technical research-paper summary.",
            ),
        ]
    )

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._llm_lock = threading.RLock()
        self._state_store = UserStateStore()
        self._extractor = PDFExtractor()
        self._chunker = SmartChunker(max_tokens=config.max_tokens_per_chunk)
        self._llm = ChatGroq(
            model=config.groq_model,
            temperature=config.groq_temperature,
            api_key=config.groq_api_key.get_secret_value(),
        )
        self._embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
        self._section_chain = self.SECTION_PROMPT | self._llm.with_structured_output(SectionSummary)
        self._final_chain = self.FINAL_PROMPT | self._llm.with_structured_output(PaperSummary)

    def summarize(self, request: SummarizeRequest) -> SummarizeResponse:
        user_state = self._state_store.get_or_create(request.user_id)

        with user_state.lock:
            raw_text, source = self._extractor.extract_text(request)
            sections = self._extractor.extract_sections(raw_text)
            chunks = self._chunker.chunk_for_summary(sections)[: self.config.max_sections]
            section_summaries = [self.summarize_chunk(chunk) for chunk in chunks]
            title = self._infer_title(raw_text)
            summary = self.generate_full_summary(title=title, section_summaries=section_summaries)
            self._state_store.increment_completed_jobs(request.user_id)

            return SummarizeResponse(
                user_id=request.user_id,
                source=source,
                sections=sections,
                chunks=chunks,
                section_summaries=section_summaries,
                summary=summary,
            )

    def summarize_batch(self, request: BatchSummarizeRequest) -> BatchSummarizeResponse:
        max_workers = min(self.config.batch_workers, len(request.requests))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            responses = list(executor.map(self.summarize, request.requests))
        return BatchSummarizeResponse(responses=responses)

    def summarize_chunk(self, chunk: SummaryChunk) -> SectionSummary:
        with self._llm_lock:
            response = self._section_chain.invoke(
                {
                    "section": chunk.section,
                    "content": chunk.content[:12000],
                }
            )

        if not isinstance(response, SectionSummary):
            raise ResearchSummarizerError("LLM returned an invalid section summary schema.")

        if response.section != chunk.section:
            return response.model_copy(update={"section": chunk.section})

        return response

    def generate_full_summary(self, title: str, section_summaries: list[SectionSummary]) -> PaperSummary:
        combined = "\n\n".join(
            f"{summary.section}: {summary.summary}" for summary in section_summaries
        )

        with self._llm_lock:
            response = self._final_chain.invoke(
                {
                    "title": title,
                    "section_summaries": combined,
                }
            )

        if not isinstance(response, PaperSummary):
            raise ResearchSummarizerError("LLM returned an invalid paper summary schema.")

        return response

    @staticmethod
    def _infer_title(raw_text: str) -> str:
        for line in raw_text.splitlines():
            cleaned = line.strip()
            if not cleaned or cleaned.upper().startswith("[PAGE"):
                continue
            if len(cleaned.split()) <= 20:
                return cleaned
        return "Research Paper"


def build_summarizer() -> ResearchSummarizer:
    return ResearchSummarizer(AppConfig())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Research Paper Summarizer using ChatGroq.")
    parser.add_argument("--pdf", type=Path, default=None, help="Path to a research PDF.")
    parser.add_argument("--sample", action="store_true", help="Use the built-in sample paper.")
    parser.add_argument("--user-id", default="default", help="Stable user/session identifier.")
    parser.add_argument(
        "--batch-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file containing a list of requests. "
            "Each item must contain user_id and either pdf_path or use_sample."
        ),
    )
    return parser.parse_args()


def _request_from_mapping(payload: object) -> SummarizeRequest:
    if not isinstance(payload, dict):
        raise UnsupportedInputError("Each batch item must be a JSON object.")

    normalized = dict(payload)
    if "pdf_path" in normalized and normalized["pdf_path"] is not None:
        normalized["pdf_path"] = Path(normalized["pdf_path"])

    return SummarizeRequest.model_validate(normalized)


def main() -> None:
    args = parse_args()
    summarizer = build_summarizer()

    if args.batch_json is not None:
        batch_path = args.batch_json.expanduser().resolve()
        with batch_path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)

        if not isinstance(payload, list):
            raise UnsupportedInputError("Batch JSON must contain a list of request objects.")

        batch_request = BatchSummarizeRequest(
            requests=[_request_from_mapping(item) for item in payload]
        )
        response = summarizer.summarize_batch(batch_request)
        print(response.model_dump_json(indent=2))
        return

    request = SummarizeRequest(
        user_id=args.user_id,
        pdf_path=args.pdf,
        use_sample=args.sample or args.pdf is None,
    )
    response = summarizer.summarize(request)
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

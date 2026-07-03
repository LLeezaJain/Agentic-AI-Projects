from __future__ import annotations
import argparse
import csv
import json
import math
import os
import random
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

load_dotenv()


class DataAnalystError(RuntimeError):
    pass


class ConfigurationError(DataAnalystError):
    pass


class DatasetNotLoadedError(DataAnalystError):
    pass


class ColumnNotFoundError(DataAnalystError):
    pass


class InvalidToolCallError(DataAnalystError):
    pass


class EmptyDatasetError(DataAnalystError):
    pass


Operator = Literal[">", "<", "==", ">=", "<=", "contains"]
AggregationFunction = Literal["mean", "sum", "count", "max", "min"]
ChartType = Literal["histogram", "bar"]
ToolName = Literal["describe_column", "filter_data", "aggregate", "correlate", "visualize"]


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    groq_api_key: SecretStr = Field(default_factory=lambda: SecretStr(os.getenv("GROQ_API_KEY", "")))
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_tool_calls: int = Field(default=5, ge=1, le=20)
    batch_workers: int = Field(default=4, ge=1, le=32)
    max_conversation_turns: int = Field(default=20, ge=1, le=200)

    @field_validator("groq_api_key")
    @classmethod
    def require_groq_api_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ConfigurationError("GROQ_API_KEY is required.")
        return value


class DatasetRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    values: dict[str, str | int | float | bool | None]


class DatasetProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rows: int = Field(ge=0)
    columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]


class LoadDatasetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str = Field(default="default", min_length=1, max_length=128)
    rows: list[DatasetRow] | None = None
    csv_path: Path | None = None

    @field_validator("csv_path")
    @classmethod
    def validate_csv_path(cls, value: Path | None) -> Path | None:
        if value is None:
            return None

        resolved = value.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"CSV file not found: {resolved}")
        if not resolved.is_file():
            raise ValueError(f"CSV path is not a file: {resolved}")
        if resolved.suffix.lower() != ".csv":
            raise ValueError(f"Only CSV input is supported: {resolved}")
        return resolved


class LoadDatasetResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str
    status: Literal["loaded"]
    profile: DatasetProfile


class DescribeColumnInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    column: str = Field(min_length=1)


class FilterDataInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    column: str = Field(min_length=1)
    operator: Operator
    value: str = Field(min_length=1)


class AggregateInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    group_by: str = Field(min_length=1)
    agg_column: str = Field(min_length=1)
    agg_func: AggregationFunction = "mean"


class CorrelateInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    col_a: str = Field(min_length=1)
    col_b: str = Field(min_length=1)


class VisualizeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    column: str = Field(min_length=1)
    chart_type: ChartType = "histogram"


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: ToolName
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reasoning: str = Field(min_length=1)
    calls: list[ToolCall] = Field(default_factory=list)


class ToolObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: ToolName
    arguments: dict[str, Any]
    result: dict[str, Any] | str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str = Field(default="default", min_length=1, max_length=128)
    query: str = Field(min_length=1)


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str
    query: str
    profile: DatasetProfile
    plan: ToolPlan
    observations: list[ToolObservation]
    answer: str


class BatchAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requests: list[AnalysisRequest] = Field(min_length=1, max_length=100)


class BatchAnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    responses: list[AnalysisResponse]


class ConversationTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class UserSession(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    user_id: str
    dataset: list[DatasetRow] = Field(default_factory=list)
    profile: DatasetProfile = Field(
        default_factory=lambda: DatasetProfile(rows=0, columns=[], numeric_columns=[], categorical_columns=[])
    )
    tool_calls_log: list[ToolObservation] = Field(default_factory=list)
    conversation: list[ConversationTurn] = Field(default_factory=list)
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

    def delete(self, user_id: str) -> None:
        with self._lock:
            if user_id not in self._sessions:
                raise KeyError(f"Unknown user_id: {user_id}")
            del self._sessions[user_id]


class DatasetLoader:
    @staticmethod
    def load(request: LoadDatasetRequest) -> list[DatasetRow]:
        if request.rows is not None:
            if not request.rows:
                raise EmptyDatasetError("Dataset rows cannot be empty.")
            return request.rows

        if request.csv_path is not None:
            with request.csv_path.open("r", encoding="utf-8-sig", newline="") as file_handle:
                reader = csv.DictReader(file_handle)
                rows = [DatasetRow(values=dict(row)) for row in reader]

            if not rows:
                raise EmptyDatasetError(f"CSV contains no rows: {request.csv_path}")
            return rows

        raise EmptyDatasetError("Provide rows or csv_path.")

    @staticmethod
    def sample_dataset() -> list[DatasetRow]:
        random.seed(42)
        categories = ["Electronics", "Clothing", "Books", "Food", "Sports"]
        regions = ["North", "South", "East", "West"]
        rows: list[DatasetRow] = []

        for index in range(50):
            category = random.choice(categories)
            region = random.choice(regions)
            units = random.randint(10, 500)
            price = round(random.uniform(5, 500), 2)
            rows.append(
                DatasetRow(
                    values={
                        "id": index + 1,
                        "category": category,
                        "region": region,
                        "units_sold": units,
                        "unit_price": price,
                        "revenue": round(units * price, 2),
                        "profit_margin": round(random.uniform(0.05, 0.45), 2),
                    }
                )
            )

        return rows


class DatasetProfiler:
    @staticmethod
    def profile(rows: list[DatasetRow]) -> DatasetProfile:
        if not rows:
            raise EmptyDatasetError("Cannot profile an empty dataset.")

        columns = list(rows[0].values.keys())
        numeric_columns: list[str] = []

        for column in columns:
            numeric_count = 0
            non_empty_count = 0

            for row in rows:
                value = row.values.get(column)
                if value is None or value == "":
                    continue
                non_empty_count += 1
                if DataAnalystTools.to_float(value) is not None:
                    numeric_count += 1

            if non_empty_count > 0 and numeric_count / non_empty_count >= 0.8:
                numeric_columns.append(column)

        categorical_columns = [column for column in columns if column not in numeric_columns]

        return DatasetProfile(
            rows=len(rows),
            columns=columns,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
        )


class DataAnalystTools:
    def __init__(self, session: UserSession) -> None:
        self.session = session

    def describe_column(self, input_model: DescribeColumnInput) -> dict[str, Any]:
        rows = self._require_dataset()
        self._require_column(input_model.column)
        values = self._numeric_values(input_model.column)

        if not values:
            raise ValueError(f"Column '{input_model.column}' has no numeric values.")

        sorted_values = sorted(values)
        quartiles = statistics.quantiles(sorted_values, n=4) if len(sorted_values) >= 4 else []

        return {
            "column": input_model.column,
            "count": len(values),
            "mean": round(statistics.mean(values), 2),
            "median": round(statistics.median(values), 2),
            "std_dev": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "range": round(max(values) - min(values), 2),
            "q1": round(quartiles[0], 2) if quartiles else None,
            "q3": round(quartiles[2], 2) if quartiles else None,
            "total_rows": len(rows),
        }

    def filter_data(self, input_model: FilterDataInput) -> dict[str, Any]:
        rows = self._require_dataset()
        self._require_column(input_model.column)

        filtered: list[dict[str, Any]] = []

        for row in rows:
            cell = row.values.get(input_model.column)
            if self._matches(cell, input_model.operator, input_model.value):
                filtered.append(row.values)

        return {
            "filter": f"{input_model.column} {input_model.operator} {input_model.value}",
            "rows_matched": len(filtered),
            "total_rows": len(rows),
            "percentage": round((len(filtered) / len(rows)) * 100, 1),
            "preview_5_rows": filtered[:5],
        }

    def aggregate(self, input_model: AggregateInput) -> dict[str, Any]:
        rows = self._require_dataset()
        self._require_column(input_model.group_by)
        self._require_column(input_model.agg_column)

        groups: dict[str, list[float]] = {}

        for row in rows:
            key = str(row.values.get(input_model.group_by, "Unknown"))
            value = self.to_float(row.values.get(input_model.agg_column))
            if value is None and input_model.agg_func != "count":
                continue
            groups.setdefault(key, [])
            if value is not None:
                groups[key].append(value)

        result: dict[str, float | int] = {}
        for key, values in groups.items():
            if input_model.agg_func == "count":
                result[key] = len(values)
            elif not values:
                continue
            elif input_model.agg_func == "mean":
                result[key] = round(statistics.mean(values), 2)
            elif input_model.agg_func == "sum":
                result[key] = round(sum(values), 2)
            elif input_model.agg_func == "max":
                result[key] = round(max(values), 2)
            elif input_model.agg_func == "min":
                result[key] = round(min(values), 2)

        sorted_result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))

        return {
            "group_by": input_model.group_by,
            "aggregation": f"{input_model.agg_func}({input_model.agg_column})",
            "groups": len(sorted_result),
            "results": sorted_result,
        }

    def correlate(self, input_model: CorrelateInput) -> dict[str, Any]:
        self._require_dataset()
        self._require_column(input_model.col_a)
        self._require_column(input_model.col_b)

        pairs: list[tuple[float, float]] = []
        for row in self.session.dataset:
            left = self.to_float(row.values.get(input_model.col_a))
            right = self.to_float(row.values.get(input_model.col_b))
            if left is not None and right is not None:
                pairs.append((left, right))

        if len(pairs) < 2:
            raise ValueError("At least two numeric pairs are required for correlation.")

        xs = [pair[0] for pair in pairs]
        ys = [pair[1] for pair in pairs]
        n = len(pairs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        covariance = sum((x - mean_x) * (y - mean_y) for x, y in pairs) / n
        std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs) / n)
        std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys) / n)

        if std_x == 0 or std_y == 0:
            raise ValueError("One column has zero variance; correlation is undefined.")

        r = covariance / (std_x * std_y)
        strength = (
            "very strong"
            if abs(r) > 0.8
            else "strong"
            if abs(r) > 0.6
            else "moderate"
            if abs(r) > 0.4
            else "weak"
            if abs(r) > 0.2
            else "very weak"
        )
        direction = "positive" if r > 0 else "negative"

        return {
            "columns": [input_model.col_a, input_model.col_b],
            "pairs": len(pairs),
            "pearson_r": round(r, 4),
            "r_squared": round(r**2, 4),
            "relationship": f"{strength} {direction} correlation",
            "interpretation": (
                f"As {input_model.col_a} increases, {input_model.col_b} tends to "
                f"{'increase' if r > 0 else 'decrease'}."
            ),
        }

    def visualize(self, input_model: VisualizeInput) -> dict[str, Any]:
        self._require_dataset()
        self._require_column(input_model.column)

        if input_model.chart_type == "histogram":
            values = self._numeric_values(input_model.column)
            if not values:
                raise ValueError(f"No numeric values found in column '{input_model.column}'.")
            return {
                "column": input_model.column,
                "chart_type": "histogram",
                "chart": self._ascii_histogram(values, input_model.column),
            }

        return {
            "column": input_model.column,
            "chart_type": "bar",
            "chart": self._ascii_bar(input_model.column),
        }

    def execute(self, call: ToolCall) -> ToolObservation:
        try:
            if call.tool == "describe_column":
                parsed = DescribeColumnInput.model_validate(call.arguments)
                result = self.describe_column(parsed)
            elif call.tool == "filter_data":
                parsed = FilterDataInput.model_validate(call.arguments)
                result = self.filter_data(parsed)
            elif call.tool == "aggregate":
                parsed = AggregateInput.model_validate(call.arguments)
                result = self.aggregate(parsed)
            elif call.tool == "correlate":
                parsed = CorrelateInput.model_validate(call.arguments)
                result = self.correlate(parsed)
            elif call.tool == "visualize":
                parsed = VisualizeInput.model_validate(call.arguments)
                result = self.visualize(parsed)
            else:
                result = {"error": f"Unsupported tool: {call.tool}"}
        except Exception as exc:
            result = {"error": f"Tool execution failed: {exc}"}

        observation = ToolObservation(tool=call.tool, arguments=call.arguments, result=result)
        self.session.tool_calls_log.append(observation)
        return observation

    def _require_dataset(self) -> list[DatasetRow]:
        if not self.session.dataset:
            raise DatasetNotLoadedError("No dataset loaded for this user session.")
        return self.session.dataset

    def _require_column(self, column: str) -> None:
        if column not in self.session.profile.columns:
            raise ColumnNotFoundError(f"Column '{column}' not found. Available columns: {self.session.profile.columns}")

    def _numeric_values(self, column: str) -> list[float]:
        values: list[float] = []
        for row in self.session.dataset:
            value = self.to_float(row.values.get(column))
            if value is not None:
                values.append(value)
        return values

    def _matches(self, cell: Any, operator: Operator, value: str) -> bool:
        if operator == "contains":
            return value.lower() in str(cell).lower()

        cell_number = self.to_float(cell)
        value_number = self.to_float(value)

        if cell_number is not None and value_number is not None:
            if operator == ">":
                return cell_number > value_number
            if operator == "<":
                return cell_number < value_number
            if operator == ">=":
                return cell_number >= value_number
            if operator == "<=":
                return cell_number <= value_number
            if operator == "==":
                return cell_number == value_number

        if operator == "==":
            return str(cell) == value

        return False

    @staticmethod
    def _ascii_histogram(values: list[float], label: str) -> str:
        n_bins = 8
        min_value = min(values)
        max_value = max(values)
        bin_size = (max_value - min_value) / n_bins if max_value != min_value else 1
        bins = [0] * n_bins

        for value in values:
            index = min(int((value - min_value) / bin_size), n_bins - 1)
            bins[index] += 1

        max_count = max(bins) if bins else 1
        width = 30
        lines = [f"Histogram: {label}", "-" * 60]

        for index, count in enumerate(bins):
            start = min_value + index * bin_size
            end = start + bin_size
            bar = "#" * int((count / max_count) * width)
            lines.append(f"{start:10.2f}-{end:10.2f} | {bar:<30} {count}")

        lines.append("-" * 60)
        return "\n".join(lines)

    def _ascii_bar(self, group_column: str) -> str:
        counts: dict[str, int] = {}
        for row in self.session.dataset:
            key = str(row.values.get(group_column, "?"))
            counts[key] = counts.get(key, 0) + 1

        max_count = max(counts.values()) if counts else 1
        width = 30
        lines = [f"Bar Chart: {group_column}", "-" * 60]

        for key, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
            bar = "#" * int((count / max_count) * width)
            lines.append(f"{key[:24]:24} | {bar:<30} {count}")

        lines.append("-" * 60)
        return "\n".join(lines)

    @staticmethod
    def to_float(value: Any) -> float | None:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return None
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(converted) or math.isinf(converted):
            return None
        return converted


class DataAnalystAgent:
    SYSTEM_PROMPT = """You are an expert data analyst AI agent.

Available tools:
- describe_column: {{"column": "<numeric column>"}}
- filter_data: {{"column": "<column>", "operator": ">|<|==|>=|<=|contains", "value": "<value>"}}
- aggregate: {{"group_by": "<column>", "agg_column": "<numeric column>", "agg_func": "mean|sum|count|max|min"}}
- correlate: {{"col_a": "<numeric column>", "col_b": "<numeric column>"}}
- visualize: {{"column": "<column>", "chart_type": "histogram|bar"}}

Plan only tool calls that are directly useful for the user's query.
Use exact column names from the dataset profile.
Do not invent columns.

IMPORTANT: The filter_data tool ONLY supports the operators '>', '<', '==', '>=', '<=', and 'contains'. It does NOT support aggregation functions like 'max', 'min', 'sum', or 'mean' as operators.
"""

    PLAN_PROMPT = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Dataset profile:\n{profile}\n\nRecent conversation:\n{conversation}\n\n"
                "User query:\n{query}\n\nReturn a tool plan.",
            ),
        ]
    )

    ANSWER_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert data analyst. Explain what the data says using the observations. "
                "Be concise, numerical, and insight-driven. Do not claim tools were used unless present in observations.",
            ),
            (
                "human",
                "Dataset profile:\n{profile}\n\nUser query:\n{query}\n\nTool plan:\n{plan}\n\n"
                "Tool observations:\n{observations}\n\nFinal answer:",
            ),
        ]
    )

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self._sessions = SessionStore()
        self._llm_lock = threading.RLock()
        self._llm = ChatGroq(
            model=self.config.groq_model,
            temperature=self.config.groq_temperature,
            groq_api_key=self.config.groq_api_key.get_secret_value(),
        )
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
        self._plan_chain = self.PLAN_PROMPT | self._llm.with_structured_output(ToolPlan)
        self._answer_chain = self.ANSWER_PROMPT | self._llm

    def load_data(self, request: LoadDatasetRequest) -> LoadDatasetResponse:
        session = self._sessions.get_or_create(request.user_id)

        with session.lock:
            rows = DatasetLoader.load(request)
            profile = DatasetProfiler.profile(rows)
            session.dataset = rows
            session.profile = profile
            session.tool_calls_log = []
            session.conversation = []

            return LoadDatasetResponse(user_id=request.user_id, status="loaded", profile=profile)

    def load_sample_data(self, user_id: str = "default") -> LoadDatasetResponse:
        return self.load_data(LoadDatasetRequest(user_id=user_id, rows=DatasetLoader.sample_dataset()))

    def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        session = self._sessions.get_or_create(request.user_id)

        with session.lock:
            if not session.dataset:
                raise DatasetNotLoadedError("Load a dataset before running analysis.")

            conversation_text = self._conversation_context(session)
            profile_json = session.profile.model_dump_json(indent=2)

            with self._llm_lock:
                plan = self._plan_chain.invoke(
                    {
                        "profile": profile_json,
                        "conversation": conversation_text,
                        "query": request.query,
                    }
                )

            if not isinstance(plan, ToolPlan):
                raise InvalidToolCallError("LLM returned an invalid tool plan schema.")

            safe_plan = plan.model_copy(update={"calls": plan.calls[: self.config.max_tool_calls]})
            tools = DataAnalystTools(session)
            observations: list[ToolObservation] = []

            for call in safe_plan.calls:
                observations.append(tools.execute(call))

            observations_json = json.dumps([observation.model_dump() for observation in observations], indent=2)

            with self._llm_lock:
                answer_message = self._answer_chain.invoke(
                    {
                        "profile": profile_json,
                        "query": request.query,
                        "plan": safe_plan.model_dump_json(indent=2),
                        "observations": observations_json,
                    }
                )

            answer = self._extract_text(answer_message)
            self._append_conversation(session, "user", request.query)
            self._append_conversation(session, "assistant", answer)

            return AnalysisResponse(
                user_id=request.user_id,
                query=request.query,
                profile=session.profile,
                plan=safe_plan,
                observations=observations,
                answer=answer,
            )

    def analyze_batch(self, request: BatchAnalysisRequest) -> BatchAnalysisResponse:
        max_workers = min(self.config.batch_workers, len(request.requests))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            responses = list(executor.map(self.analyze, request.requests))
        return BatchAnalysisResponse(responses=responses)

    def delete_user_session(self, user_id: str) -> None:
        self._sessions.delete(user_id)

    def _append_conversation(self, session: UserSession, role: Literal["user", "assistant"], content: str) -> None:
        session.conversation.append(ConversationTurn(role=role, content=content))
        if len(session.conversation) > self.config.max_conversation_turns:
            session.conversation = session.conversation[-self.config.max_conversation_turns :]

    @staticmethod
    def _conversation_context(session: UserSession) -> str:
        if not session.conversation:
            return "(none)"
        return "\n".join(f"{turn.role.upper()}: {turn.content}" for turn in session.conversation[-10:])

    @staticmethod
    def _extract_text(response: object) -> str:
        content = getattr(response, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        raise DataAnalystError("LLM returned an empty or unsupported response.")


def load_rows_from_json(path: Path) -> list[DatasetRow]:
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    if not isinstance(payload, list):
        raise ValueError("JSON dataset must be a list of objects.")

    rows: list[DatasetRow] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Every JSON dataset row must be an object.")
        rows.append(DatasetRow(values=item))

    if not rows:
        raise EmptyDatasetError("JSON dataset contains no rows.")

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Data Analyst Agent using ChatGroq.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    load_parser = subparsers.add_parser("load")
    load_parser.add_argument("--user-id", default="default")
    load_parser.add_argument("--csv", type=Path, default=None)
    load_parser.add_argument("--json", type=Path, default=None)
    load_parser.add_argument("--sample", action="store_true")

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("query")
    analyze_parser.add_argument("--user-id", default="default")
    analyze_parser.add_argument("--sample", action="store_true")
    analyze_parser.add_argument("--csv", type=Path, default=None)
    analyze_parser.add_argument("--json", type=Path, default=None)

    batch_parser = subparsers.add_parser("batch")
    batch_parser.add_argument("batch_json", type=Path)
    batch_parser.add_argument("--user-id", default="default")
    batch_parser.add_argument("--sample", action="store_true")
    batch_parser.add_argument("--csv", type=Path, default=None)
    batch_parser.add_argument("--json", type=Path, default=None)

    demo_parser = subparsers.add_parser("demo")
    demo_parser.add_argument("--user-id", default="demo-user")

    return parser.parse_args()


def load_initial_dataset(agent: DataAnalystAgent, args: argparse.Namespace) -> None:
    if getattr(args, "sample", False):
        agent.load_sample_data(user_id=args.user_id)
        return

    csv_path = getattr(args, "csv", None)
    if csv_path is not None:
        agent.load_data(LoadDatasetRequest(user_id=args.user_id, csv_path=csv_path))
        return

    json_path = getattr(args, "json", None)
    if json_path is not None:
        agent.load_data(LoadDatasetRequest(user_id=args.user_id, rows=load_rows_from_json(json_path)))
        return


def main() -> None:
    args = parse_args()
    agent = DataAnalystAgent()

    if args.command == "load":
        if args.sample:
            response = agent.load_sample_data(user_id=args.user_id)
        elif args.csv is not None:
            response = agent.load_data(LoadDatasetRequest(user_id=args.user_id, csv_path=args.csv))
        elif args.json is not None:
            response = agent.load_data(
                LoadDatasetRequest(user_id=args.user_id, rows=load_rows_from_json(args.json))
            )
        else:
            raise EmptyDatasetError("Use --sample, --csv, or --json.")
        print(response.model_dump_json(indent=2))
        return

    if args.command == "analyze":
        load_initial_dataset(agent, args)
        response = agent.analyze(AnalysisRequest(user_id=args.user_id, query=args.query))
        print(response.model_dump_json(indent=2))
        return

    if args.command == "batch":
        load_initial_dataset(agent, args)
        batch_path = args.batch_json.expanduser().resolve()
        with batch_path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)

        if not isinstance(payload, list):
            raise ValueError("Batch JSON must contain a list of query strings or request objects.")

        requests: list[AnalysisRequest] = []
        for item in payload:
            if isinstance(item, str):
                requests.append(AnalysisRequest(user_id=args.user_id, query=item))
            elif isinstance(item, dict):
                normalized = {"user_id": args.user_id, **item}
                requests.append(AnalysisRequest.model_validate(normalized))
            else:
                raise ValueError("Batch items must be query strings or request objects.")

        response = agent.analyze_batch(BatchAnalysisRequest(requests=requests))
        print(response.model_dump_json(indent=2))
        return

    if args.command == "demo":
        agent.load_sample_data(user_id=args.user_id)
        queries = [
            "Give me summary statistics for revenue and units_sold.",
            "Which category has the highest average revenue?",
            "Is there a correlation between units_sold and revenue?",
            "Show me a distribution chart of revenue.",
            "Filter products where revenue is greater than 10000.",
        ]
        response = agent.analyze_batch(
            BatchAnalysisRequest(
                requests=[AnalysisRequest(user_id=args.user_id, query=query) for query in queries]
            )
        )
        print(response.model_dump_json(indent=2))
        return

    raise DataAnalystError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()

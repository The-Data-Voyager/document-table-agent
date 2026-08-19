"""Conservative natural-English translation into validated table queries."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any

import pandas as pd

from app.agent.query_engine import QueryResult, TableQueryAgent
from app.agent.request_parser import (
    AggregationSpec,
    FilterSpec,
    QueryRequest,
    SortSpec,
)


class QuestionInterpretationError(ValueError):
    """Base error for an English question that cannot be translated safely."""


class UnsupportedQuestionError(QuestionInterpretationError):
    """Raised when the question does not identify a supported metric."""


class AmbiguousQuestionError(QuestionInterpretationError):
    """Raised when multiple plausible interpretations remain."""


def _non_empty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _normalized_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).replace("’", "'").lower()
    normalized = re.sub(r"(?<=[a-z0-9])'s\b", "", normalized)
    normalized = re.sub(r"[^a-z0-9&_-]+", " ", normalized)
    return " ".join(normalized.split())


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(
        re.search(
            rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])",
            text,
        )
    )


@dataclass(frozen=True)
class MetricSemantic:
    """Natural-language names and query behavior for one numeric metric."""

    name: str
    table: str
    column: str
    aliases: tuple[str, ...]
    default_aggregation: str
    display_columns: tuple[str, ...]
    date_column: str | None = None

    def __post_init__(self) -> None:
        name = _non_empty_string(self.name, field_name="metric name")
        table = _non_empty_string(self.table, field_name="metric table")
        column = _non_empty_string(self.column, field_name="metric column")
        aliases = tuple(
            _normalized_text(
                _non_empty_string(alias, field_name="metric alias")
            )
            for alias in self.aliases
        )
        if not aliases or len(aliases) != len(set(aliases)):
            raise ValueError("metric aliases must be non-empty and unique.")
        if self.default_aggregation not in {"sum", "mean", "min", "max"}:
            raise ValueError(
                "default_aggregation must be sum, mean, min, or max."
            )
        display_columns = tuple(self.display_columns)
        if column not in display_columns:
            raise ValueError("display_columns must contain the metric column.")
        date_column = self.date_column
        if date_column is not None:
            date_column = _non_empty_string(
                date_column,
                field_name="metric date_column",
            )
            if date_column not in display_columns:
                raise ValueError(
                    "display_columns must contain the metric date_column."
                )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "table", table)
        object.__setattr__(self, "column", column)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "display_columns", display_columns)
        object.__setattr__(self, "date_column", date_column)


@dataclass(frozen=True)
class DimensionSemantic:
    """Natural-language aliases for one grouping/filter column."""

    column: str
    aliases: tuple[str, ...]

    def __post_init__(self) -> None:
        column = _non_empty_string(self.column, field_name="dimension column")
        aliases = tuple(
            _normalized_text(
                _non_empty_string(alias, field_name="dimension alias")
            )
            for alias in self.aliases
        )
        if not aliases or len(aliases) != len(set(aliases)):
            raise ValueError("dimension aliases must be non-empty and unique.")
        object.__setattr__(self, "column", column)
        object.__setattr__(self, "aliases", aliases)


@dataclass(frozen=True)
class TotalExclusion:
    """Reported total row that must not be double-counted in aggregations."""

    table: str
    column: str
    value: Any

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "table",
            _non_empty_string(self.table, field_name="exclusion table"),
        )
        object.__setattr__(
            self,
            "column",
            _non_empty_string(self.column, field_name="exclusion column"),
        )


@dataclass(frozen=True)
class SemanticCatalog:
    """Profile vocabulary used by the generic English question parser."""

    name: str
    metrics: tuple[MetricSemantic, ...]
    dimensions: tuple[DimensionSemantic, ...]
    total_exclusions: tuple[TotalExclusion, ...] = ()

    def __post_init__(self) -> None:
        name = _non_empty_string(self.name, field_name="catalog name")
        metrics = tuple(self.metrics)
        dimensions = tuple(self.dimensions)
        exclusions = tuple(self.total_exclusions)
        if not metrics or any(not isinstance(item, MetricSemantic) for item in metrics):
            raise ValueError("metrics must contain at least one MetricSemantic.")
        if not dimensions or any(
            not isinstance(item, DimensionSemantic) for item in dimensions
        ):
            raise ValueError(
                "dimensions must contain at least one DimensionSemantic."
            )
        if any(not isinstance(item, TotalExclusion) for item in exclusions):
            raise TypeError("total_exclusions must contain TotalExclusion items.")
        metric_names = tuple(item.name for item in metrics)
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("metric names must be unique.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "dimensions", dimensions)
        object.__setattr__(self, "total_exclusions", exclusions)


@dataclass(frozen=True)
class QuestionInterpretation:
    """The transparent translation produced for one English question."""

    question: str
    metric_name: str
    request: QueryRequest
    notes: tuple[str, ...]


@dataclass(frozen=True)
class NaturalLanguageQueryResult:
    """English interpretation paired with its executed query result."""

    interpretation: QuestionInterpretation
    query_result: QueryResult

    @property
    def answer(self) -> pd.DataFrame:
        return self.query_result.answer

    @property
    def evidence(self) -> pd.DataFrame:
        return self.query_result.evidence


_DATE_PATTERN = re.compile(
    r"(?<!\d)(\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4})(?!\d)"
)
_AGGREGATION_CUES = {
    "mean": ("average", "avg", "mean"),
    "sum": ("total", "sum", "combined"),
    "max": ("maximum", "max"),
    "min": ("minimum", "min"),
    "median": ("median",),
}
_DESCENDING_RANK_CUES = ("highest", "largest", "greatest", "most", "top")
_ASCENDING_RANK_CUES = ("lowest", "smallest", "least", "fewest", "bottom")


def _metric_match(
    normalized_question: str,
    catalog: SemanticCatalog,
) -> tuple[MetricSemantic, str]:
    occurrences: list[tuple[MetricSemantic, str, int, int]] = []
    for metric in catalog.metrics:
        for alias in metric.aliases:
            for match in re.finditer(
                rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
                normalized_question,
            ):
                occurrences.append(
                    (metric, alias, match.start(), match.end())
                )
    active = []
    for occurrence in occurrences:
        _metric, _alias, start, end = occurrence
        contained_by_longer = any(
            other_start <= start
            and other_end >= end
            and (other_end - other_start) > (end - start)
            for _other_metric, _other_alias, other_start, other_end in occurrences
        )
        if not contained_by_longer:
            active.append(occurrence)
    matches: dict[str, tuple[MetricSemantic, str]] = {}
    for metric, alias, _start, _end in active:
        previous = matches.get(metric.name)
        if previous is None or len(alias) > len(previous[1]):
            matches[metric.name] = (metric, alias)
    if not matches:
        supported = sorted(metric.name for metric in catalog.metrics)
        raise UnsupportedQuestionError(
            "The question does not identify a supported metric. "
            f"Supported metrics: {supported!r}."
        )
    if len(matches) > 1:
        raise AmbiguousQuestionError(
            "The question mentions multiple metrics: "
            f"{sorted(matches)!r}. Ask about one metric at a time."
        )
    return next(iter(matches.values()))


def _without_metric_aliases(text: str, metric: MetricSemantic) -> str:
    result = text
    for alias in sorted(metric.aliases, key=len, reverse=True):
        result = re.sub(
            rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
            " ",
            result,
        )
    return " ".join(result.split())


def _explicit_aggregation(text: str) -> str | None:
    matches = {
        operation
        for operation, cues in _AGGREGATION_CUES.items()
        if any(_contains_phrase(text, cue) for cue in cues)
    }
    if len(matches) > 1:
        raise AmbiguousQuestionError(
            f"Conflicting aggregation words were found: {sorted(matches)!r}."
        )
    return next(iter(matches)) if matches else None


def _ranking_direction(text: str) -> str | None:
    descending = any(
        _contains_phrase(text, cue) for cue in _DESCENDING_RANK_CUES
    )
    ascending = any(
        _contains_phrase(text, cue) for cue in _ASCENDING_RANK_CUES
    )
    if descending and ascending:
        raise AmbiguousQuestionError(
            "The question requests both highest and lowest ordering."
        )
    if descending:
        return "desc"
    if ascending:
        return "asc"
    return None


def _result_limit(text: str, ranking_direction: str | None) -> int:
    match = re.search(r"\b(?:top|bottom)\s+(\d+)\b", text)
    if match:
        limit = int(match.group(1))
        if not 1 <= limit <= 1_000:
            raise QuestionInterpretationError(
                "A requested top/bottom count must be between 1 and 1000."
            )
        return limit
    if ranking_direction is not None and _contains_phrase(text, "which"):
        return 1
    return 100


def _dimension_requested(
    text: str,
    dimension: DimensionSemantic,
    ranking_direction: str | None,
) -> bool:
    for alias in dimension.aliases:
        escaped = re.escape(alias)
        if re.search(
            rf"\b(?:by|per|each|which)\s+(?:the\s+)?{escaped}\b",
            text,
        ):
            return True
        if ranking_direction is not None and _contains_phrase(text, alias):
            return True
    return False


def _parse_date(value: str) -> str:
    date_format = "%Y-%m-%d" if value[:4].isdigit() and value[4] == "-" else "%d-%m-%Y"
    try:
        return datetime.strptime(value, date_format).date().isoformat()
    except ValueError as error:
        raise QuestionInterpretationError(
            f"Invalid date in question: {value!r}."
        ) from error


def _date_filter(
    question: str,
    date_column: str | None,
) -> tuple[FilterSpec, ...]:
    matches = _DATE_PATTERN.findall(question)
    if len(matches) > 2:
        raise AmbiguousQuestionError(
            "More than two dates were found; use one date or one date range."
        )
    dates = tuple(_parse_date(value) for value in matches)
    if dates and date_column is None:
        raise UnsupportedQuestionError(
            "This metric does not define a date column for date filtering."
        )
    if len(dates) == 2:
        if dates[0] > dates[1]:
            raise QuestionInterpretationError(
                "The date range starts after it ends."
            )
        return (FilterSpec(date_column, "between", dates),)
    if len(dates) == 1:
        return (FilterSpec(date_column, "eq", dates[0]),)
    return ()


def _value_matches_question(
    question: str,
    normalized_question: str,
    value: str,
) -> bool:
    normalized_value = _normalized_text(value)
    if not normalized_value or normalized_value == "all india":
        return False
    if len(normalized_value) <= 2:
        return bool(
            re.search(
                rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])",
                question,
            )
        )
    return _contains_phrase(normalized_question, normalized_value)


def _entity_filters(
    question: str,
    normalized_question: str,
    table: pd.DataFrame,
    dimensions: tuple[DimensionSemantic, ...],
) -> tuple[tuple[FilterSpec, ...], dict[str, tuple[Any, ...]]]:
    filters: list[FilterSpec] = []
    matches_by_column: dict[str, tuple[Any, ...]] = {}
    for dimension in dimensions:
        if dimension.column not in table.columns:
            continue
        values = tuple(
            value
            for value in table[dimension.column].dropna().unique().tolist()
            if isinstance(value, str)
            and _value_matches_question(
                question,
                normalized_question,
                value,
            )
        )
        if not values:
            continue
        matches_by_column[dimension.column] = values
        operator = "eq" if len(values) == 1 else "in"
        filter_value: Any = values[0] if len(values) == 1 else values
        filters.append(FilterSpec(dimension.column, operator, filter_value))
    return tuple(filters), matches_by_column


def _aggregation_alias(operation: str, metric: MetricSemantic) -> str:
    labels = {
        "sum": "Total",
        "mean": "Average",
        "median": "Median",
        "min": "Minimum",
        "max": "Maximum",
    }
    return f"{labels[operation]}_{metric.column}"


def interpret_question(
    question: str,
    catalog: SemanticCatalog,
    tables: Mapping[str, pd.DataFrame],
) -> QuestionInterpretation:
    """Translate one supported English question into a ``QueryRequest``."""

    question = _non_empty_string(question, field_name="question")
    if not isinstance(catalog, SemanticCatalog):
        raise TypeError("catalog must be a SemanticCatalog.")
    normalized = _normalized_text(question)
    metric, matched_alias = _metric_match(normalized, catalog)
    if metric.table not in tables:
        raise UnsupportedQuestionError(
            f"Metric {metric.name!r} requires unavailable table "
            f"{metric.table!r}."
        )
    table = tables[metric.table]
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"Table {metric.table!r} is not a DataFrame.")
    missing_display = set(metric.display_columns) - set(table.columns)
    if missing_display:
        raise UnsupportedQuestionError(
            f"Table {metric.table!r} is missing semantic columns: "
            f"{sorted(missing_display)!r}."
        )

    intent_text = _without_metric_aliases(normalized, metric)
    explicit_aggregation = _explicit_aggregation(intent_text)
    ranking_direction = _ranking_direction(intent_text)
    limit = _result_limit(intent_text, ranking_direction)
    requested_dimensions = tuple(
        dimension.column
        for dimension in catalog.dimensions
        if dimension.column in table.columns
        and _dimension_requested(intent_text, dimension, ranking_direction)
    )
    entity_filters, matched_entities = _entity_filters(
        question,
        normalized,
        table,
        catalog.dimensions,
    )
    comparison_requested = _contains_phrase(intent_text, "compare")
    if comparison_requested and sum(map(len, matched_entities.values())) < 2:
        raise AmbiguousQuestionError(
            "A comparison requires at least two recognized states or regions."
        )

    group_by = list(requested_dimensions)
    if comparison_requested and explicit_aggregation is not None:
        for dimension in catalog.dimensions:
            if (
                dimension.column in matched_entities
                and dimension.column not in group_by
            ):
                group_by.append(dimension.column)

    aggregate_requested = bool(
        group_by or explicit_aggregation is not None or ranking_direction is not None
    )
    filters = list(entity_filters)
    filters.extend(_date_filter(question, metric.date_column))
    notes = [
        f"Matched metric phrase {matched_alias!r} to {metric.column!r}."
    ]

    if aggregate_requested:
        operation = explicit_aggregation or metric.default_aggregation
        alias = _aggregation_alias(operation, metric)
        for exclusion in catalog.total_exclusions:
            if exclusion.table == metric.table:
                filters.append(
                    FilterSpec(exclusion.column, "ne", exclusion.value)
                )
                notes.append(
                    f"Excluded reported total {exclusion.value!r} before aggregation."
                )
        if ranking_direction is not None:
            direction = ranking_direction
        elif group_by:
            direction = "desc"
        else:
            direction = None
        request = QueryRequest(
            table=metric.table,
            filters=tuple(filters),
            group_by=tuple(group_by),
            aggregations=(
                AggregationSpec(metric.column, operation, alias),
            ),
            order_by=(SortSpec(alias, direction),) if direction else (),
            limit=limit,
        )
        notes.append(
            f"Used {operation!r} aggregation"
            + (f" grouped by {group_by!r}." if group_by else ".")
        )
    else:
        order_columns = []
        for dimension in catalog.dimensions:
            if (
                dimension.column in metric.display_columns
                and dimension.column not in order_columns
            ):
                order_columns.append(dimension.column)
        if (
            metric.date_column is not None
            and metric.date_column not in order_columns
        ):
            order_columns.append(metric.date_column)
        request = QueryRequest(
            table=metric.table,
            columns=metric.display_columns,
            filters=tuple(filters),
            order_by=tuple(SortSpec(column, "asc") for column in order_columns),
            limit=limit,
        )
        notes.append("Returned matching source rows without aggregation.")

    return QuestionInterpretation(
        question=question,
        metric_name=metric.name,
        request=request,
        notes=tuple(notes),
    )


class NaturalLanguageTableAgent:
    """Facade combining semantic interpretation with deterministic execution."""

    def __init__(
        self,
        tables: Mapping[str, pd.DataFrame],
        catalog: SemanticCatalog,
    ) -> None:
        snapshots = {
            name: table.copy(deep=True) for name, table in tables.items()
        }
        self._tables = MappingProxyType(snapshots)
        self._catalog = catalog
        self._query_agent = TableQueryAgent(snapshots)

    def interpret(self, question: str) -> QuestionInterpretation:
        """Translate a question without executing it."""

        return interpret_question(question, self._catalog, self._tables)

    def ask(self, question: str) -> NaturalLanguageQueryResult:
        """Interpret and execute one supported English question."""

        interpretation = self.interpret(question)
        query_result = self._query_agent.execute(interpretation.request)
        return NaturalLanguageQueryResult(interpretation, query_result)

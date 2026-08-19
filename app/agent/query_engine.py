"""Safe, generic execution of validated table-query requests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import pandas as pd

from app.agent.request_parser import (
    AggregationSpec,
    FilterSpec,
    QueryRequest,
    parse_query_request,
)


class QueryExecutionError(ValueError):
    """Base error for a structurally valid request that cannot be executed."""


class UnknownTableError(QueryExecutionError):
    """Raised when a request names a table outside the registry."""


class UnknownColumnError(QueryExecutionError):
    """Raised when a request names a column outside the relevant schema."""


class NoMatchingRowsError(QueryExecutionError):
    """Raised when filters match no source rows."""


@dataclass(frozen=True)
class QueryResult:
    """Answer rows plus the filtered source rows supporting the result."""

    request: QueryRequest
    answer: pd.DataFrame
    evidence: pd.DataFrame
    source_row_count: int
    matched_row_count: int

    @property
    def returned_row_count(self) -> int:
        return len(self.answer)


def _column_names(columns: pd.Index) -> set[Any]:
    return set(columns.tolist())


def _require_columns(
    requested: Sequence[str],
    available: pd.Index,
    *,
    context: str,
) -> None:
    missing = set(requested) - _column_names(available)
    if missing:
        raise UnknownColumnError(
            f"Unknown {context} columns: {sorted(missing)!r}. Available "
            f"columns: {available.tolist()!r}."
        )


def _coerce_scalar(series: pd.Series, value: Any) -> Any:
    if value is None:
        return None
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        try:
            return pd.to_datetime(value, errors="raise")
        except (TypeError, ValueError) as error:
            raise QueryExecutionError(
                f"Could not interpret {value!r} as a date for "
                f"column {series.name!r}."
            ) from error
    if pd.api.types.is_numeric_dtype(series.dtype):
        try:
            return pd.to_numeric(value, errors="raise")
        except (TypeError, ValueError) as error:
            raise QueryExecutionError(
                f"Could not interpret {value!r} as numeric for "
                f"column {series.name!r}."
            ) from error
    return value


def _filter_mask(series: pd.Series, spec: FilterSpec) -> pd.Series:
    operator = spec.operator
    if operator == "contains":
        return series.astype("string").str.contains(
            spec.value,
            regex=False,
            na=False,
        )
    if operator in {"in", "not_in", "between"}:
        values = tuple(_coerce_scalar(series, item) for item in spec.value)
    else:
        value = _coerce_scalar(series, spec.value)

    if operator == "eq":
        return series.isna() if value is None else series.eq(value).fillna(False)
    if operator == "ne":
        return series.notna() if value is None else series.ne(value).fillna(False)
    if operator in {"gt", "gte", "lt", "lte"} and value is None:
        raise QueryExecutionError(
            f"Filter {operator!r} cannot compare column {series.name!r} to null."
        )
    if operator == "gt":
        return series.gt(value).fillna(False)
    if operator == "gte":
        return series.ge(value).fillna(False)
    if operator == "lt":
        return series.lt(value).fillna(False)
    if operator == "lte":
        return series.le(value).fillna(False)
    if operator == "in":
        return series.isin(values)
    if operator == "not_in":
        return ~series.isin(values)
    if operator == "between":
        return series.between(values[0], values[1], inclusive="both").fillna(False)
    raise AssertionError(f"Unhandled filter operator: {operator!r}.")


def _apply_filters(
    source: pd.DataFrame,
    filters: tuple[FilterSpec, ...],
) -> pd.DataFrame:
    result = source.copy(deep=True)
    for spec in filters:
        mask = _filter_mask(result[spec.column], spec)
        result = result.loc[mask].copy()
    return result


def _validate_aggregation_dtype(
    series: pd.Series,
    spec: AggregationSpec,
) -> None:
    if spec.operation in {"sum", "mean", "median"} and not (
        pd.api.types.is_numeric_dtype(series.dtype)
    ):
        raise QueryExecutionError(
            f"Aggregation {spec.operation!r} requires a numeric column; "
            f"{spec.column!r} has dtype {series.dtype}."
        )


def _aggregate_without_groups(
    source: pd.DataFrame,
    aggregations: tuple[AggregationSpec, ...],
) -> pd.DataFrame:
    values: dict[str, list[Any]] = {}
    for spec in aggregations:
        series = source[spec.column]
        _validate_aggregation_dtype(series, spec)
        if spec.operation == "count":
            result = series.count()
        elif spec.operation == "nunique":
            result = series.nunique(dropna=True)
        else:
            result = getattr(series, spec.operation)()
        values[spec.alias] = [result]
    return pd.DataFrame(values)


def _aggregate_with_groups(
    source: pd.DataFrame,
    group_by: tuple[str, ...],
    aggregations: tuple[AggregationSpec, ...],
) -> pd.DataFrame:
    for spec in aggregations:
        _validate_aggregation_dtype(source[spec.column], spec)
    named_aggregations = {
        spec.alias: pd.NamedAgg(column=spec.column, aggfunc=spec.operation)
        for spec in aggregations
    }
    return (
        source.groupby(list(group_by), dropna=False, sort=False)
        .agg(**named_aggregations)
        .reset_index()
    )


def execute_query(
    tables: Mapping[str, pd.DataFrame],
    request: QueryRequest,
) -> QueryResult:
    """Execute one validated request without mutating the table registry."""

    if not isinstance(request, QueryRequest):
        raise TypeError("request must be a QueryRequest.")
    if request.table not in tables:
        raise UnknownTableError(
            f"Unknown table {request.table!r}. Available tables: "
            f"{sorted(tables)!r}."
        )
    source = tables[request.table]
    if not isinstance(source, pd.DataFrame):
        raise TypeError(f"Registered table {request.table!r} is not a DataFrame.")
    if isinstance(source.columns, pd.MultiIndex):
        raise QueryExecutionError(
            "The query engine requires flat analysis-ready column names."
        )
    source_snapshot = source.copy(deep=True)

    filter_columns = tuple(item.column for item in request.filters)
    aggregation_columns = tuple(item.column for item in request.aggregations)
    _require_columns(
        request.columns + request.group_by + filter_columns + aggregation_columns,
        source.columns,
        context="source",
    )

    filtered = _apply_filters(source, request.filters)
    if filtered.empty:
        raise NoMatchingRowsError("The request filters matched no source rows.")
    evidence = filtered.reset_index(drop=True).copy(deep=True)

    if request.aggregations:
        if request.group_by:
            answer = _aggregate_with_groups(
                filtered,
                request.group_by,
                request.aggregations,
            )
        else:
            answer = _aggregate_without_groups(filtered, request.aggregations)
    else:
        selected_columns = (
            list(request.columns) if request.columns else source.columns.tolist()
        )
        answer = filtered.loc[:, selected_columns].copy(deep=True)

    sort_columns = tuple(item.column for item in request.order_by)
    _require_columns(sort_columns, answer.columns, context="result-ordering")
    if request.order_by:
        answer = answer.sort_values(
            by=list(sort_columns),
            ascending=[item.direction == "asc" for item in request.order_by],
            kind="mergesort",
        )
    answer = answer.head(request.limit).reset_index(drop=True)

    pd.testing.assert_frame_equal(source, source_snapshot, check_exact=True)
    return QueryResult(
        request=request,
        answer=answer,
        evidence=evidence,
        source_row_count=len(source),
        matched_row_count=len(filtered),
    )


class TableQueryAgent:
    """Small facade combining strict request parsing and query execution."""

    def __init__(self, tables: Mapping[str, pd.DataFrame]) -> None:
        if not isinstance(tables, Mapping) or not tables:
            raise ValueError("tables must be a non-empty mapping.")
        registry: dict[str, pd.DataFrame] = {}
        for name, table in tables.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Every table name must be a non-empty string.")
            if not isinstance(table, pd.DataFrame):
                raise TypeError(f"Registered table {name!r} is not a DataFrame.")
            normalized_name = name.strip()
            if normalized_name in registry:
                raise ValueError(
                    f"Duplicate normalized table name: {normalized_name!r}."
                )
            registry[normalized_name] = table.copy(deep=True)
        self._tables = MappingProxyType(registry)

    def describe_tables(self) -> dict[str, dict[str, Any]]:
        """Return compact schemas callers can use to construct requests."""

        return {
            name: {
                "rows": len(table),
                "columns": table.columns.tolist(),
                "dtypes": {
                    str(column): str(dtype)
                    for column, dtype in table.dtypes.items()
                },
            }
            for name, table in self._tables.items()
        }

    def ask(self, payload: Mapping[str, Any] | str) -> QueryResult:
        """Parse and execute one deterministic request."""

        return self.execute(parse_query_request(payload))

    def execute(self, request: QueryRequest) -> QueryResult:
        """Execute an already parsed request against the registered tables."""

        return execute_query(self._tables, request)

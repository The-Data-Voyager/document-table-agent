"""Strict parsing for deterministic table-query requests.

The parser accepts mappings or JSON objects.  It deliberately does not attempt
to interpret unrestricted natural language; that can be added later as a
separate adapter which must produce the same validated ``QueryRequest``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


ALLOWED_FILTER_OPERATORS = frozenset(
    {
        "eq",
        "ne",
        "gt",
        "gte",
        "lt",
        "lte",
        "in",
        "not_in",
        "between",
        "contains",
    }
)
ALLOWED_AGGREGATIONS = frozenset(
    {"sum", "mean", "median", "min", "max", "count", "nunique"}
)
ALLOWED_SORT_DIRECTIONS = frozenset({"asc", "desc"})
MAX_RESULT_ROWS = 1_000


def _non_empty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _string_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a list of column names.")
    result = tuple(
        _non_empty_string(item, field_name=f"{field_name} item")
        for item in value
    )
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} cannot contain duplicates.")
    return result


def _mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object.")
    return value


def _reject_unknown_keys(
    value: Mapping[str, Any],
    allowed: set[str],
    *,
    field_name: str,
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            f"{field_name} contains unsupported fields: {sorted(unknown)!r}."
        )


@dataclass(frozen=True)
class FilterSpec:
    """One allowlisted comparison applied to a source column."""

    column: str
    operator: str
    value: Any

    def __post_init__(self) -> None:
        column = _non_empty_string(self.column, field_name="filter column")
        operator = _non_empty_string(
            self.operator,
            field_name="filter operator",
        ).lower()
        if operator not in ALLOWED_FILTER_OPERATORS:
            raise ValueError(
                f"Unsupported filter operator {operator!r}; expected one of "
                f"{sorted(ALLOWED_FILTER_OPERATORS)!r}."
            )
        if operator in {"in", "not_in", "between"}:
            if isinstance(self.value, (str, bytes)) or not isinstance(
                self.value,
                Sequence,
            ):
                raise TypeError(
                    f"Filter operator {operator!r} requires a list value."
                )
            values = tuple(self.value)
            if operator == "between" and len(values) != 2:
                raise ValueError("A between filter requires exactly two values.")
            if operator in {"in", "not_in"} and not values:
                raise ValueError(f"An {operator} filter cannot use an empty list.")
            object.__setattr__(self, "value", values)
        if operator == "contains" and not isinstance(self.value, str):
            raise TypeError("A contains filter requires a string value.")
        object.__setattr__(self, "column", column)
        object.__setattr__(self, "operator", operator)


@dataclass(frozen=True)
class AggregationSpec:
    """One named aggregation over a source column."""

    column: str
    operation: str
    alias: str | None = None

    def __post_init__(self) -> None:
        column = _non_empty_string(
            self.column,
            field_name="aggregation column",
        )
        operation = _non_empty_string(
            self.operation,
            field_name="aggregation operation",
        ).lower()
        if operation not in ALLOWED_AGGREGATIONS:
            raise ValueError(
                f"Unsupported aggregation {operation!r}; expected one of "
                f"{sorted(ALLOWED_AGGREGATIONS)!r}."
            )
        alias = (
            f"{operation}_{column}"
            if self.alias is None
            else _non_empty_string(self.alias, field_name="aggregation alias")
        )
        object.__setattr__(self, "column", column)
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "alias", alias)


@dataclass(frozen=True)
class SortSpec:
    """One stable result-ordering rule."""

    column: str
    direction: str = "asc"

    def __post_init__(self) -> None:
        column = _non_empty_string(self.column, field_name="sort column")
        direction = _non_empty_string(
            self.direction,
            field_name="sort direction",
        ).lower()
        if direction not in ALLOWED_SORT_DIRECTIONS:
            raise ValueError("sort direction must be 'asc' or 'desc'.")
        object.__setattr__(self, "column", column)
        object.__setattr__(self, "direction", direction)


@dataclass(frozen=True)
class QueryRequest:
    """Validated, deterministic request understood by the query engine."""

    table: str
    columns: tuple[str, ...] = ()
    filters: tuple[FilterSpec, ...] = ()
    group_by: tuple[str, ...] = ()
    aggregations: tuple[AggregationSpec, ...] = ()
    order_by: tuple[SortSpec, ...] = ()
    limit: int = 100

    def __post_init__(self) -> None:
        table = _non_empty_string(self.table, field_name="table")
        columns = _string_tuple(self.columns, field_name="columns")
        filters = tuple(self.filters)
        group_by = _string_tuple(self.group_by, field_name="group_by")
        aggregations = tuple(self.aggregations)
        order_by = tuple(self.order_by)

        if any(not isinstance(item, FilterSpec) for item in filters):
            raise TypeError("filters must contain FilterSpec instances.")
        if any(not isinstance(item, AggregationSpec) for item in aggregations):
            raise TypeError("aggregations must contain AggregationSpec instances.")
        if any(not isinstance(item, SortSpec) for item in order_by):
            raise TypeError("order_by must contain SortSpec instances.")
        if group_by and not aggregations:
            raise ValueError("group_by requires at least one aggregation.")
        if columns and aggregations:
            raise ValueError(
                "columns cannot be combined with aggregations; aggregated "
                "output columns are determined by group_by and aliases."
            )
        aliases = tuple(item.alias for item in aggregations)
        if len(aliases) != len(set(aliases)):
            raise ValueError("aggregation aliases must be unique.")
        if set(group_by) & set(aliases):
            raise ValueError(
                "aggregation aliases cannot duplicate group_by columns."
            )
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise TypeError("limit must be an integer.")
        if not 1 <= self.limit <= MAX_RESULT_ROWS:
            raise ValueError(
                f"limit must be between 1 and {MAX_RESULT_ROWS}."
            )

        object.__setattr__(self, "table", table)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "filters", filters)
        object.__setattr__(self, "group_by", group_by)
        object.__setattr__(self, "aggregations", aggregations)
        object.__setattr__(self, "order_by", order_by)


def _parse_filter(value: Any) -> FilterSpec:
    item = _mapping(value, field_name="filter")
    _reject_unknown_keys(
        item,
        {"column", "operator", "value"},
        field_name="filter",
    )
    missing = {"column", "operator", "value"} - set(item)
    if missing:
        raise ValueError(f"filter is missing fields: {sorted(missing)!r}.")
    return FilterSpec(
        column=item["column"],
        operator=item["operator"],
        value=item["value"],
    )


def _parse_aggregation(value: Any) -> AggregationSpec:
    item = _mapping(value, field_name="aggregation")
    _reject_unknown_keys(
        item,
        {"column", "operation", "alias"},
        field_name="aggregation",
    )
    missing = {"column", "operation"} - set(item)
    if missing:
        raise ValueError(f"aggregation is missing fields: {sorted(missing)!r}.")
    return AggregationSpec(
        column=item["column"],
        operation=item["operation"],
        alias=item.get("alias"),
    )


def _parse_sort(value: Any) -> SortSpec:
    item = _mapping(value, field_name="order_by item")
    _reject_unknown_keys(
        item,
        {"column", "direction"},
        field_name="order_by item",
    )
    if "column" not in item:
        raise ValueError("order_by item is missing field: 'column'.")
    return SortSpec(
        column=item["column"],
        direction=item.get("direction", "asc"),
    )


def parse_query_request(payload: Mapping[str, Any] | str) -> QueryRequest:
    """Parse a JSON object or mapping into an immutable ``QueryRequest``."""

    if isinstance(payload, str):
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("Request is not valid JSON.") from error
    else:
        decoded = payload
    request = _mapping(decoded, field_name="request")
    allowed = {
        "table",
        "columns",
        "filters",
        "group_by",
        "aggregations",
        "order_by",
        "limit",
    }
    _reject_unknown_keys(request, allowed, field_name="request")
    if "table" not in request:
        raise ValueError("request is missing field: 'table'.")

    filters_value = request.get("filters", ())
    aggregations_value = request.get("aggregations", ())
    order_by_value = request.get("order_by", ())
    for value, name in (
        (filters_value, "filters"),
        (aggregations_value, "aggregations"),
        (order_by_value, "order_by"),
    ):
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"{name} must be a list.")

    return QueryRequest(
        table=request["table"],
        columns=_string_tuple(request.get("columns"), field_name="columns"),
        filters=tuple(_parse_filter(item) for item in filters_value),
        group_by=_string_tuple(
            request.get("group_by"),
            field_name="group_by",
        ),
        aggregations=tuple(
            _parse_aggregation(item) for item in aggregations_value
        ),
        order_by=tuple(_parse_sort(item) for item in order_by_value),
        limit=request.get("limit", 100),
    )

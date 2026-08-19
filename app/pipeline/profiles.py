"""Immutable configuration models for profile-driven table pipelines."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


def _normalized_positions(
    positions: tuple[int, ...],
    *,
    name: str,
    allow_empty: bool = True,
) -> tuple[int, ...]:
    normalized = tuple(positions)
    if not allow_empty and not normalized:
        raise ValueError(f"{name} cannot be empty.")
    if any(isinstance(position, bool) or not isinstance(position, int) for position in normalized):
        raise TypeError(f"{name} must contain only integers.")
    if any(position < 0 for position in normalized):
        raise ValueError(f"{name} cannot contain negative positions.")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} cannot contain duplicate positions.")
    return normalized


def _normalized_terms(terms: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    normalized = tuple(term.strip() for term in terms)
    if not normalized or any(not term for term in normalized):
        raise ValueError(f"{name} must contain at least one non-empty term.")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} cannot contain duplicate terms.")
    return normalized


@dataclass(frozen=True)
class ColumnMappingProfile:
    """Explicit rule for assigning one column from another column's values."""

    key_column_position: int
    target_column_position: int
    values: Mapping[Any, Any]
    strict: bool = True

    def __post_init__(self) -> None:
        positions = _normalized_positions(
            (self.key_column_position, self.target_column_position),
            name="mapping column positions",
            allow_empty=False,
        )
        if positions[0] == positions[1]:
            raise ValueError("Mapping key and target columns must differ.")
        if not isinstance(self.values, Mapping):
            raise TypeError("values must implement the Mapping interface.")
        if not isinstance(self.strict, bool):
            raise TypeError("strict must be a boolean.")
        object.__setattr__(
            self,
            "values",
            MappingProxyType(dict(self.values)),
        )


@dataclass(frozen=True)
class TableTransformationProfile:
    """Declarative transformation choices for one table layout."""

    header_row_positions: tuple[int, ...]
    identity_column_positions: tuple[int, ...] = ()
    measure_column_positions: tuple[int, ...] = ()
    fill_merged_headers_from_column: int | None = None
    forward_fill_column_positions: tuple[int, ...] = ()
    remove_devanagari: bool = False
    normalize_text_whitespace: bool = True
    mapping: ColumnMappingProfile | None = None
    preprocessor: Callable[[Any], Any] | None = None

    def __post_init__(self) -> None:
        headers = _normalized_positions(
            self.header_row_positions,
            name="header_row_positions",
            allow_empty=False,
        )
        identities = _normalized_positions(
            self.identity_column_positions,
            name="identity_column_positions",
        )
        measures = _normalized_positions(
            self.measure_column_positions,
            name="measure_column_positions",
        )
        fill_columns = _normalized_positions(
            self.forward_fill_column_positions,
            name="forward_fill_column_positions",
        )
        if set(identities) & set(measures):
            raise ValueError(
                "Identity and measure column positions cannot overlap."
            )
        if headers != tuple(sorted(headers)):
            raise ValueError("header_row_positions must be ascending.")
        if len(headers) > 1 and any(
            later != earlier + 1
            for earlier, later in zip(headers, headers[1:])
        ):
            raise ValueError("Multi-row header positions must be consecutive.")
        if (
            self.fill_merged_headers_from_column is not None
            and len(headers) == 1
        ):
            raise ValueError(
                "fill_merged_headers_from_column requires a multi-row header."
            )
        if self.fill_merged_headers_from_column is not None and (
            isinstance(self.fill_merged_headers_from_column, bool)
            or not isinstance(self.fill_merged_headers_from_column, int)
            or self.fill_merged_headers_from_column < 0
        ):
            raise TypeError(
                "fill_merged_headers_from_column must be a non-negative integer."
            )
        if not isinstance(self.remove_devanagari, bool):
            raise TypeError("remove_devanagari must be a boolean.")
        if not isinstance(self.normalize_text_whitespace, bool):
            raise TypeError("normalize_text_whitespace must be a boolean.")
        if self.mapping is not None and not isinstance(
            self.mapping,
            ColumnMappingProfile,
        ):
            raise TypeError("mapping must be a ColumnMappingProfile or None.")
        if self.preprocessor is not None and not callable(self.preprocessor):
            raise TypeError("preprocessor must be callable or None.")

        object.__setattr__(self, "header_row_positions", headers)
        object.__setattr__(self, "identity_column_positions", identities)
        object.__setattr__(self, "measure_column_positions", measures)
        object.__setattr__(self, "forward_fill_column_positions", fill_columns)


@dataclass(frozen=True)
class OutputSchemaProfile:
    """Expected structural contract after transformation."""

    column_count: int | None = None
    column_levels: int = 1
    expected_row_count: int | None = None
    required_columns: tuple[Any, ...] = ()
    non_null_columns: tuple[Any, ...] = ()
    unique_key_columns: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if self.column_count is not None and (
            isinstance(self.column_count, bool)
            or not isinstance(self.column_count, int)
            or self.column_count < 1
        ):
            raise TypeError("column_count must be a positive integer or None.")
        if (
            isinstance(self.column_levels, bool)
            or not isinstance(self.column_levels, int)
            or self.column_levels < 1
        ):
            raise TypeError("column_levels must be a positive integer.")
        if self.expected_row_count is not None and (
            isinstance(self.expected_row_count, bool)
            or not isinstance(self.expected_row_count, int)
            or self.expected_row_count < 0
        ):
            raise TypeError(
                "expected_row_count must be a non-negative integer or None."
            )

        for field_name in (
            "required_columns",
            "non_null_columns",
            "unique_key_columns",
        ):
            values = tuple(getattr(self, field_name))
            try:
                has_duplicates = len(values) != len(set(values))
            except TypeError as error:
                raise TypeError(
                    f"{field_name} values must be hashable column labels."
                ) from error
            if has_duplicates:
                raise ValueError(f"{field_name} cannot contain duplicates.")
            object.__setattr__(self, field_name, values)


@dataclass(frozen=True)
class TablePostprocessorProfile:
    """Optional named output derived from a transformed wide table."""

    name: str
    processor: Callable[[Any], Any]
    output_schema: OutputSchemaProfile
    output_filename: str | None = None

    def __post_init__(self) -> None:
        normalized_name = self.name.strip()
        if not normalized_name:
            raise ValueError("Postprocessor name cannot be empty.")
        if not callable(self.processor):
            raise TypeError("processor must be callable.")
        if not isinstance(self.output_schema, OutputSchemaProfile):
            raise TypeError("output_schema must be an OutputSchemaProfile.")
        if self.output_filename is not None:
            filename = self.output_filename.strip()
            if not filename.lower().endswith(".csv"):
                raise ValueError("output_filename must use the .csv extension.")
            object.__setattr__(self, "output_filename", filename)
        object.__setattr__(self, "name", normalized_name)


@dataclass(frozen=True)
class TableProfile:
    """Search, selection, transformation, and output rules for one table."""

    name: str
    search_terms: tuple[str, ...]
    transformation: TableTransformationProfile
    table_index: int = 0
    page_match_index: int = 0
    page_end_search_terms: tuple[str, ...] = ()
    include_end_page: bool = False
    output_filename: str | None = None
    output_schema: OutputSchemaProfile | None = None
    postprocessors: tuple[TablePostprocessorProfile, ...] = ()

    def __post_init__(self) -> None:
        normalized_name = self.name.strip()
        if not normalized_name:
            raise ValueError("Table profile name cannot be empty.")
        terms = _normalized_terms(self.search_terms, name="search_terms")
        if isinstance(self.table_index, bool) or not isinstance(self.table_index, int):
            raise TypeError("table_index must be an integer.")
        if self.table_index < 0:
            raise ValueError("table_index cannot be negative.")
        if isinstance(self.page_match_index, bool) or not isinstance(
            self.page_match_index,
            int,
        ):
            raise TypeError("page_match_index must be an integer.")
        if self.page_match_index < 0:
            raise ValueError("page_match_index cannot be negative.")
        end_terms = tuple(self.page_end_search_terms)
        if end_terms:
            end_terms = _normalized_terms(
                end_terms,
                name="page_end_search_terms",
            )
        if not isinstance(self.include_end_page, bool):
            raise TypeError("include_end_page must be a boolean.")
        if self.include_end_page and not end_terms:
            raise ValueError(
                "include_end_page requires page_end_search_terms."
            )
        if not isinstance(self.transformation, TableTransformationProfile):
            raise TypeError(
                "transformation must be a TableTransformationProfile."
            )
        if self.output_filename is not None:
            filename = self.output_filename.strip()
            if not filename.lower().endswith(".csv"):
                raise ValueError("output_filename must use the .csv extension.")
            object.__setattr__(self, "output_filename", filename)
        if self.output_schema is not None and not isinstance(
            self.output_schema,
            OutputSchemaProfile,
        ):
            raise TypeError(
                "output_schema must be an OutputSchemaProfile or None."
            )
        postprocessors = tuple(self.postprocessors)
        if any(
            not isinstance(postprocessor, TablePostprocessorProfile)
            for postprocessor in postprocessors
        ):
            raise TypeError(
                "postprocessors must contain TablePostprocessorProfile instances."
            )
        postprocessor_names = [item.name for item in postprocessors]
        if len(postprocessor_names) != len(set(postprocessor_names)):
            raise ValueError("Postprocessor names must be unique per table.")

        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "search_terms", terms)
        object.__setattr__(self, "page_end_search_terms", end_terms)
        object.__setattr__(self, "postprocessors", postprocessors)


@dataclass(frozen=True)
class DocumentProfile:
    """Detection markers and table profiles for one document family."""

    name: str
    detection_terms: tuple[str, ...]
    tables: tuple[TableProfile, ...]

    def __post_init__(self) -> None:
        normalized_name = self.name.strip()
        if not normalized_name:
            raise ValueError("Document profile name cannot be empty.")
        terms = _normalized_terms(
            self.detection_terms,
            name="detection_terms",
        )
        tables = tuple(self.tables)
        if not tables:
            raise ValueError("A document profile must contain at least one table.")
        if any(not isinstance(table, TableProfile) for table in tables):
            raise TypeError("tables must contain only TableProfile instances.")
        table_names = [table.name for table in tables]
        if len(table_names) != len(set(table_names)):
            raise ValueError("Table profile names must be unique per document.")

        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "detection_terms", terms)
        object.__setattr__(self, "tables", tables)

"""Explicit, non-mutating transformations for raw Pandas tables.

This module is intentionally separate from extraction and validation. Every
public function returns a new DataFrame and requires the caller to choose the
rows or columns to transform; validation hints are never applied implicitly.
"""

from __future__ import annotations

from collections.abc import Mapping
from operator import index
import re
from typing import Any, Iterable

import pandas as pd


_DEVANAGARI_PATTERN = re.compile(
    "[\u0900-\u097F\u1CD0-\u1CFF\uA8E0-\uA8FF"
    "\U00011B00-\U00011B5F]+"
)
_EMPTY_DELIMITER_PATTERNS = (
    re.compile(r"\(\s*\)"),
    re.compile(r"\[\s*\]"),
    re.compile(r"\{\s*\}"),
)
_ORPHAN_LEADING_PERIODS = re.compile(r"^(?:\.\s*){2,}")


def _ensure_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")


def _is_missing(value: Any) -> bool:
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _remove_devanagari_from_value(
    value: Any,
    *,
    normalize_whitespace: bool,
) -> Any:
    if not isinstance(value, str):
        return value

    cleaned = _DEVANAGARI_PATTERN.sub("", value)
    cleaned = cleaned.replace("\u200c", "").replace("\u200d", "")
    for delimiter_pattern in _EMPTY_DELIMITER_PATTERNS:
        cleaned = delimiter_pattern.sub("", cleaned)
    cleaned = _ORPHAN_LEADING_PERIODS.sub("", cleaned)

    if normalize_whitespace:
        return " ".join(cleaned.split())
    return cleaned.strip()


def _validate_positions(
    positions: Iterable[int],
    *,
    upper_bound: int,
    name: str,
) -> list[int]:
    try:
        supplied_positions = list(positions)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of integers.") from error

    normalized = []
    for position in supplied_positions:
        if isinstance(position, bool):
            raise TypeError(f"{name} must contain only integers.")
        try:
            normalized_position = index(position)
        except TypeError as error:
            raise TypeError(
                f"{name} must contain only integers."
            ) from error
        if normalized_position < 0 or normalized_position >= upper_bound:
            raise IndexError(
                f"{name} position {normalized_position} is outside the "
                f"valid range 0 to {upper_bound - 1}."
            )
        normalized.append(normalized_position)

    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} cannot contain duplicate positions.")

    return normalized


def promote_single_header_row(
    df: pd.DataFrame,
    header_row_position: int,
) -> pd.DataFrame:
    """Promote one raw row to columns and drop all rows through that header.

    Header values are preserved exactly. Missing or blank header cells cause
    an error instead of being silently renamed.
    """

    _ensure_dataframe(df)
    header_positions = _validate_positions(
        [header_row_position],
        upper_bound=df.shape[0],
        name="header_row_position",
    )
    header_position = header_positions[0]
    header_values = df.iloc[header_position].tolist()
    missing_columns = [
        position
        for position, value in enumerate(header_values)
        if _is_missing(value)
    ]
    if missing_columns:
        raise ValueError(
            "Single-row header contains missing or blank cells at column "
            f"positions {missing_columns}."
        )

    transformed = df.iloc[header_position + 1 :].copy(deep=True)
    transformed.columns = pd.Index(header_values, tupleize_cols=False)
    return transformed.reset_index(drop=True)


def promote_multirow_header(
    df: pd.DataFrame,
    header_row_positions: Iterable[int],
    *,
    fill_merged_from_column: int | None = None,
) -> pd.DataFrame:
    """Promote consecutive rows to a source-preserving MultiIndex header.

    ``fill_merged_from_column`` can be used to forward-fill horizontal merged
    labels within each header row, beginning at the explicitly supplied
    column position. Columns before that position are left untouched.
    """

    _ensure_dataframe(df)
    header_positions = _validate_positions(
        header_row_positions,
        upper_bound=df.shape[0],
        name="header_row_positions",
    )
    if len(header_positions) < 2:
        raise ValueError(
            "promote_multirow_header requires at least two header rows."
        )
    if header_positions != sorted(header_positions):
        raise ValueError("header_row_positions must be in ascending order.")
    if any(
        later != earlier + 1
        for earlier, later in zip(
            header_positions,
            header_positions[1:],
        )
    ):
        raise ValueError("Multi-row header positions must be consecutive.")

    column_count = df.shape[1]
    if fill_merged_from_column is not None:
        if isinstance(fill_merged_from_column, bool):
            raise TypeError("fill_merged_from_column must be an integer.")
        try:
            fill_start = index(fill_merged_from_column)
        except TypeError as error:
            raise TypeError(
                "fill_merged_from_column must be an integer."
            ) from error
        if fill_start < 0 or fill_start >= column_count:
            raise IndexError(
                "fill_merged_from_column is outside the valid column range."
            )
    else:
        fill_start = None

    header_levels = []
    for row_position in header_positions:
        level = [
            "" if _is_missing(value) else value
            for value in df.iloc[row_position].tolist()
        ]
        if fill_start is not None:
            previous_value: Any = ""
            for column_position in range(fill_start, column_count):
                if level[column_position] == "" and previous_value != "":
                    level[column_position] = previous_value
                elif level[column_position] != "":
                    previous_value = level[column_position]
        header_levels.append(level)

    column_tuples = list(zip(*header_levels))
    transformed = df.iloc[header_positions[-1] + 1 :].copy(deep=True)
    transformed.columns = pd.MultiIndex.from_tuples(column_tuples)
    return transformed.reset_index(drop=True)


def forward_fill_columns(
    df: pd.DataFrame,
    column_positions: Iterable[int],
    *,
    treat_blank_as_missing: bool = True,
) -> pd.DataFrame:
    """Return a copy with only the selected columns forward-filled."""

    _ensure_dataframe(df)
    positions = _validate_positions(
        column_positions,
        upper_bound=df.shape[1],
        name="column_positions",
    )
    transformed = df.copy(deep=True)

    for column_position in positions:
        values = transformed.iloc[:, column_position].copy()
        if treat_blank_as_missing:
            values = values.map(
                lambda value: (
                    pd.NA
                    if isinstance(value, str) and not value.strip()
                    else value
                )
            )
        transformed.isetitem(column_position, values.ffill())

    return transformed


def find_unmapped_key_values(
    df: pd.DataFrame,
    key_column_position: int,
    mapping: Mapping[Any, Any],
) -> list[Any]:
    """Return distinct non-missing key values absent from ``mapping``."""

    _ensure_dataframe(df)
    if not isinstance(mapping, Mapping):
        raise TypeError("mapping must implement the Mapping interface.")
    key_position = _validate_positions(
        [key_column_position],
        upper_bound=df.shape[1],
        name="key_column_position",
    )[0]
    unmapped = []

    for value in df.iloc[:, key_position].tolist():
        if _is_missing(value):
            continue
        try:
            is_mapped = value in mapping
        except TypeError:
            is_mapped = False
        if not is_mapped and value not in unmapped:
            unmapped.append(value)

    return unmapped


def assign_values_from_mapping(
    df: pd.DataFrame,
    *,
    key_column_position: int,
    target_column_position: int,
    mapping: Mapping[Any, Any],
    strict: bool = True,
) -> pd.DataFrame:
    """Assign a target column from an explicit key-to-value mapping.

    Rows with a missing key retain a populated target, which supports total
    rows such as ``ALL INDIA``. In strict mode, any non-missing unmapped key,
    or a row with both key and target missing, raises an error rather than
    silently guessing.
    """

    _ensure_dataframe(df)
    if not isinstance(mapping, Mapping):
        raise TypeError("mapping must implement the Mapping interface.")
    if not isinstance(strict, bool):
        raise TypeError("strict must be a boolean.")
    key_position = _validate_positions(
        [key_column_position],
        upper_bound=df.shape[1],
        name="key_column_position",
    )[0]
    target_position = _validate_positions(
        [target_column_position],
        upper_bound=df.shape[1],
        name="target_column_position",
    )[0]

    transformed = df.copy(deep=True)
    target_values = transformed.iloc[:, target_position].copy()
    unresolved = []

    for row_position in range(len(transformed)):
        key = transformed.iloc[row_position, key_position]
        current_target = target_values.iloc[row_position]

        if _is_missing(key):
            if _is_missing(current_target):
                unresolved.append((row_position, key))
            continue

        try:
            is_mapped = key in mapping
        except TypeError:
            is_mapped = False

        if is_mapped:
            target_values.iloc[row_position] = mapping[key]
        else:
            unresolved.append((row_position, key))

    if strict and unresolved:
        raise ValueError(
            "Unmapped or missing key values at row positions: "
            f"{unresolved!r}."
        )

    transformed.isetitem(target_position, target_values)
    return transformed


def convert_columns_to_numeric(
    df: pd.DataFrame,
    column_positions: Iterable[int],
    *,
    errors: str = "raise",
) -> pd.DataFrame:
    """Return a copy with explicitly selected columns converted to numeric."""

    _ensure_dataframe(df)
    if errors not in {"raise", "coerce"}:
        raise ValueError("errors must be either 'raise' or 'coerce'.")
    positions = _validate_positions(
        column_positions,
        upper_bound=df.shape[1],
        name="column_positions",
    )
    transformed = df.copy(deep=True)

    for column_position in positions:
        converted = pd.to_numeric(
            transformed.iloc[:, column_position],
            errors=errors,
        )
        transformed.isetitem(column_position, converted)

    return transformed


def remove_devanagari_text(
    df: pd.DataFrame,
    *,
    include_column_labels: bool = True,
    normalize_whitespace: bool = True,
) -> pd.DataFrame:
    """Remove Devanagari characters from a copied table.

    English text, punctuation with meaningful content, numeric values, and
    non-string objects are preserved. Empty delimiters left by removed text
    are discarded. Whitespace is collapsed by default so bilingual cells
    become clean single-line English values.
    """

    _ensure_dataframe(df)
    if not isinstance(include_column_labels, bool):
        raise TypeError("include_column_labels must be a boolean.")
    if not isinstance(normalize_whitespace, bool):
        raise TypeError("normalize_whitespace must be a boolean.")

    transformed = df.copy(deep=True)
    for column_position in range(transformed.shape[1]):
        cleaned_values = transformed.iloc[:, column_position].map(
            lambda value: _remove_devanagari_from_value(
                value,
                normalize_whitespace=normalize_whitespace,
            )
        )
        transformed.isetitem(column_position, cleaned_values)

    if not include_column_labels:
        return transformed

    def clean_label(label: Any) -> Any:
        return _remove_devanagari_from_value(
            label,
            normalize_whitespace=normalize_whitespace,
        )

    if isinstance(transformed.columns, pd.MultiIndex):
        cleaned_columns = [
            tuple(clean_label(level_value) for level_value in column)
            for column in transformed.columns.tolist()
        ]
        cleaned_names = [
            clean_label(name) for name in transformed.columns.names
        ]
        transformed.columns = pd.MultiIndex.from_tuples(
            cleaned_columns,
            names=cleaned_names,
        )
    else:
        transformed.columns = pd.Index(
            [clean_label(label) for label in transformed.columns],
            name=clean_label(transformed.columns.name),
            tupleize_cols=False,
        )

    return transformed


def transform_table(
    df: pd.DataFrame,
    *,
    header_row_positions: Iterable[int],
    fill_merged_headers_from_column: int | None = None,
    forward_fill_column_positions: Iterable[int] = (),
    numeric_column_positions: Iterable[int] = (),
    numeric_errors: str = "raise",
    remove_devanagari: bool = False,
    normalize_text_whitespace: bool = True,
) -> pd.DataFrame:
    """Apply an explicit header, fill, and numeric-conversion pipeline.

    Column positions refer to the table after its header has been promoted.
    The input remains unchanged at every stage.
    """

    _ensure_dataframe(df)
    if not isinstance(remove_devanagari, bool):
        raise TypeError("remove_devanagari must be a boolean.")
    if not isinstance(normalize_text_whitespace, bool):
        raise TypeError("normalize_text_whitespace must be a boolean.")
    header_positions = _validate_positions(
        header_row_positions,
        upper_bound=df.shape[0],
        name="header_row_positions",
    )
    if not header_positions:
        raise ValueError("At least one header row position is required.")

    if len(header_positions) == 1:
        if fill_merged_headers_from_column is not None:
            raise ValueError(
                "fill_merged_headers_from_column applies only to multi-row "
                "headers."
            )
        transformed = promote_single_header_row(
            df,
            header_positions[0],
        )
    else:
        transformed = promote_multirow_header(
            df,
            header_positions,
            fill_merged_from_column=fill_merged_headers_from_column,
        )

    transformed = forward_fill_columns(
        transformed,
        forward_fill_column_positions,
    )
    transformed = convert_columns_to_numeric(
        transformed,
        numeric_column_positions,
        errors=numeric_errors,
    )
    if remove_devanagari:
        transformed = remove_devanagari_text(
            transformed,
            normalize_whitespace=normalize_text_whitespace,
        )
    return transformed

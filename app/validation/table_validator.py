"""Read-only validation helpers for raw tabular data.

The functions in this module inspect a :class:`pandas.DataFrame` and return
metadata about possible quality or structure issues. They deliberately do not
clean, fill, rename, reshape, or otherwise modify the supplied table.

Row references in validation results are zero-based row positions. Column
references are the original column labels when those labels are unique. If
labels are duplicated, a ``(position, label)`` tuple is used so that no
column is hidden in the report.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from numbers import Number
from typing import Any

import pandas as pd


_NUMERIC_DATE_PATTERN = re.compile(
    r"(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|"
    r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2})"
)
_NAMED_MONTH_DATE_PATTERN = re.compile(
    r"\d{1,2}(?:-|\s)(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|"
    r"Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
    r"Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"(?:-|\s)\d{2,4}",
    flags=re.IGNORECASE,
)
_DATE_FORMATS = (
    "%d-%m-%Y",
    "%d-%m-%y",
    "%d/%m/%Y",
    "%d/%m/%y",
    "%d.%m.%Y",
    "%d.%m.%y",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%d-%b-%Y",
    "%d-%b-%y",
    "%d %b %Y",
    "%d %b %y",
    "%d-%B-%Y",
    "%d-%B-%y",
    "%d %B %Y",
    "%d %B %y",
)
_NUMBER_PATTERN = re.compile(
    r"[+-]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)"
)


def _ensure_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")


def _is_blank_string(value: Any) -> bool:
    return isinstance(value, str) and not value.strip()


def _missing_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new mask treating nulls and blank strings as missing."""

    _ensure_dataframe(df)
    blank_values = [
        [_is_blank_string(value) for value in row]
        for row in df.to_numpy(dtype=object, copy=False)
    ]
    blank_mask = pd.DataFrame(
        blank_values,
        index=df.index,
        columns=df.columns,
    )
    return df.isna() | blank_mask


def _column_reference(df: pd.DataFrame, position: int) -> Any:
    label = df.columns[position]
    if df.columns.is_unique:
        return label
    return (position, label)


def _is_date_like(value: Any) -> bool:
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return not pd.isna(value)

    # NumPy datetime64 values are common in otherwise object-typed tables.
    if type(value).__name__ == "datetime64":
        return not pd.isna(value)

    if not isinstance(value, str):
        return False

    candidate = value.strip()
    if not candidate:
        return False

    if not (
        _NUMERIC_DATE_PATTERN.fullmatch(candidate)
        or _NAMED_MONTH_DATE_PATTERN.fullmatch(candidate)
    ):
        return False

    for date_format in _DATE_FORMATS:
        try:
            datetime.strptime(candidate, date_format)
        except ValueError:
            continue
        return True

    return False


def _is_number_like(value: Any) -> bool:
    if isinstance(value, Number) and not isinstance(value, bool):
        return not pd.isna(value)
    if not isinstance(value, str):
        return False
    return bool(_NUMBER_PATTERN.fullmatch(value.strip()))


def _row_profile(
    df: pd.DataFrame,
    row_position: int,
    missing_mask: pd.DataFrame,
) -> dict[str, int]:
    populated = 0
    dates = 0
    numbers = 0
    text = 0

    for column_position in range(df.shape[1]):
        if bool(missing_mask.iloc[row_position, column_position]):
            continue

        populated += 1
        value = df.iloc[row_position, column_position]
        if _is_date_like(value):
            dates += 1
        elif _is_number_like(value):
            numbers += 1
        elif isinstance(value, str):
            text += 1

    return {
        "populated": populated,
        "dates": dates,
        "numbers": numbers,
        "text": text,
    }


def get_dataframe_dimensions(df: pd.DataFrame) -> dict[str, int]:
    """Return the number of rows and columns in ``df``."""

    _ensure_dataframe(df)
    return {"rows": int(df.shape[0]), "columns": int(df.shape[1])}


def count_missing_cells(df: pd.DataFrame) -> int:
    """Count null cells and strings containing only whitespace."""

    return int(_missing_mask(df).to_numpy().sum())


def count_missing_by_column(df: pd.DataFrame) -> dict[Any, int]:
    """Return missing-cell counts for each column."""

    mask = _missing_mask(df)
    return {
        _column_reference(df, position): int(mask.iloc[:, position].sum())
        for position in range(df.shape[1])
    }


def find_empty_rows(df: pd.DataFrame) -> list[int]:
    """Return zero-based positions of completely empty rows."""

    mask = _missing_mask(df)
    return [
        position
        for position, is_empty in enumerate(mask.all(axis=1).tolist())
        if bool(is_empty)
    ]


def find_empty_columns(df: pd.DataFrame) -> list[Any]:
    """Return references for columns in which every cell is empty."""

    mask = _missing_mask(df)
    return [
        _column_reference(df, position)
        for position, is_empty in enumerate(mask.all(axis=0).tolist())
        if bool(is_empty)
    ]


def find_duplicate_rows(df: pd.DataFrame) -> list[int]:
    """Return positions of exact duplicate rows after their first occurrence."""

    _ensure_dataframe(df)
    duplicate_mask = df.duplicated(keep="first").tolist()
    return [
        position
        for position, is_duplicate in enumerate(duplicate_mask)
        if bool(is_duplicate)
    ]


def count_non_empty_cells_by_row(df: pd.DataFrame) -> list[int]:
    """Return a populated-cell count for each row position."""

    mask = _missing_mask(df)
    return [int(value) for value in (~mask).sum(axis=1).tolist()]


def detect_title_like_rows(
    df: pd.DataFrame,
    *,
    scan_rows: int = 5,
    max_populated_cells: int | None = None,
) -> list[int]:
    """Return leading row positions that may contain a table title.

    The heuristic looks only near the beginning of a table.  A candidate must
    contain text in a small, contiguous group of populated cells and be
    followed by a denser row.  This remains an intentionally cautious hint,
    not a classification guarantee.
    """

    _ensure_dataframe(df)
    if scan_rows < 1:
        raise ValueError("scan_rows must be at least 1.")
    if max_populated_cells is not None and max_populated_cells < 1:
        raise ValueError("max_populated_cells must be at least 1.")

    row_count, column_count = df.shape
    if row_count == 0 or column_count < 2:
        return []

    if max_populated_cells is None:
        max_populated_cells = max(1, min(3, column_count // 4))

    mask = _missing_mask(df)
    non_empty_counts = count_non_empty_cells_by_row(df)
    scan_limit = min(row_count, scan_rows)
    possible_titles = []

    for row_position in range(scan_limit):
        populated = non_empty_counts[row_position]
        if populated == 0 or populated > max_populated_cells:
            continue

        populated_positions = [
            column_position
            for column_position in range(column_count)
            if not bool(mask.iloc[row_position, column_position])
        ]
        cells_are_contiguous = (
            populated_positions[-1] - populated_positions[0] + 1
            == populated
        )
        if not cells_are_contiguous:
            continue

        contains_text = any(
            isinstance(df.iloc[row_position, column_position], str)
            and not _is_date_like(
                df.iloc[row_position, column_position]
            )
            and not _is_number_like(
                df.iloc[row_position, column_position]
            )
            for column_position in populated_positions
        )
        if not contains_text:
            continue

        later_counts = [
            count
            for count in non_empty_counts[row_position + 1 : scan_limit]
            if count > 0
        ]
        if later_counts and later_counts[0] <= populated:
            continue

        possible_titles.append(row_position)

    return possible_titles


def find_date_like_cells(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Locate cells that conservatively match a valid date value or pattern."""

    _ensure_dataframe(df)
    mask = _missing_mask(df)
    matches = []

    for row_position in range(df.shape[0]):
        for column_position in range(df.shape[1]):
            if bool(mask.iloc[row_position, column_position]):
                continue
            value = df.iloc[row_position, column_position]
            if _is_date_like(value):
                matches.append(
                    {
                        "row": row_position,
                        "column": _column_reference(
                            df,
                            column_position,
                        ),
                        "value": value,
                    }
                )

    return matches


def detect_possible_header_rows(
    df: pd.DataFrame,
    *,
    scan_rows: int = 5,
    title_rows: list[int] | None = None,
) -> list[int]:
    """Return leading rows with cautious header-like signals.

    Date labels, text followed by numeric-looking data, and sparse grouped
    labels followed by a denser row are treated as possible signals.  Rows
    already identified as title-like are excluded.
    """

    _ensure_dataframe(df)
    if scan_rows < 1:
        raise ValueError("scan_rows must be at least 1.")

    row_count, column_count = df.shape
    if row_count == 0 or column_count == 0:
        return []

    if title_rows is None:
        title_rows = detect_title_like_rows(df, scan_rows=scan_rows)

    excluded_titles = set(title_rows)
    mask = _missing_mask(df)
    scan_limit = min(row_count, scan_rows)
    profiles = [
        _row_profile(df, row_position, mask)
        for row_position in range(scan_limit)
    ]
    possible_headers = []

    for row_position, profile in enumerate(profiles):
        if row_position in excluded_titles or profile["populated"] == 0:
            continue

        later_profiles = [
            later_profile
            for later_profile in profiles[row_position + 1 :]
            if later_profile["populated"] > 0
        ]
        next_profile = later_profiles[0] if later_profiles else None

        contains_dates = profile["dates"] > 0
        text_precedes_numeric_data = (
            profile["text"] > 0
            and profile["numbers"] == 0
            and next_profile is not None
            and next_profile["numbers"] > 0
        )
        sparse_group_labels = (
            profile["text"] >= 2
            and profile["numbers"] == 0
            and next_profile is not None
            and profile["populated"] < next_profile["populated"]
        )

        if (
            contains_dates
            or text_precedes_numeric_data
            or sparse_group_labels
        ):
            possible_headers.append(row_position)

    return possible_headers


def detect_possible_hierarchical_header(
    df: pd.DataFrame,
    *,
    header_rows: list[int] | None = None,
) -> bool:
    """Return whether adjacent leading rows may form a multi-row header."""

    _ensure_dataframe(df)
    if header_rows is None:
        header_rows = detect_possible_header_rows(df)

    ordered_rows = sorted(set(header_rows))
    return any(
        later == earlier + 1
        for earlier, later in zip(ordered_rows, ordered_rows[1:])
    )


def detect_merged_cell_like_columns(df: pd.DataFrame) -> list[Any]:
    """Find columns with internal missing runs consistent with merged cells.

    A column is reported only when a missing cell occurs between two populated
    cells.  This conservative pattern is often produced by vertically merged
    group labels, but ordinary missing data can produce it too.
    """

    _ensure_dataframe(df)
    mask = _missing_mask(df)
    possible_columns = []

    for column_position in range(df.shape[1]):
        column_missing = [
            bool(value)
            for value in mask.iloc[:, column_position].tolist()
        ]
        populated_positions = [
            position
            for position, is_missing in enumerate(column_missing)
            if not is_missing
        ]
        if len(populated_positions) < 2:
            continue

        first_populated = populated_positions[0]
        last_populated = populated_positions[-1]
        has_internal_gap = any(
            column_missing[first_populated + 1 : last_populated]
        )
        if has_internal_gap:
            possible_columns.append(
                _column_reference(df, column_position)
            )

    return possible_columns


def _build_warnings(
    *,
    missing_cells: int,
    duplicate_rows: int,
    empty_rows: list[int],
    empty_columns: list[Any],
    title_rows: list[int],
    header_rows: list[int],
    date_cell_count: int,
    hierarchical_header: bool,
    merged_cell_columns: list[Any],
) -> list[str]:
    warnings = []

    if missing_cells:
        warnings.append(
            f"{missing_cells} missing or blank cells detected."
        )
    if empty_rows:
        warnings.append(
            f"Completely empty row positions detected: {empty_rows}."
        )
    if empty_columns:
        warnings.append(
            "Completely empty columns detected: "
            f"{empty_columns!r}."
        )
    if duplicate_rows:
        warnings.append(
            f"{duplicate_rows} duplicate rows detected after their first "
            "occurrence."
        )
    if title_rows:
        warnings.append(
            f"Rows {title_rows} may be title rows; detection is heuristic."
        )
    if header_rows:
        warnings.append(
            f"Rows {header_rows} may contain header content; detection is "
            "heuristic."
        )
    if date_cell_count:
        warnings.append(
            f"{date_cell_count} date-like cells detected; they may be "
            "headers or data values."
        )
    if hierarchical_header:
        warnings.append(
            "Adjacent header-like rows suggest a possible multi-row or "
            "hierarchical header; this is not guaranteed."
        )
    if merged_cell_columns:
        warnings.append(
            "Columns with internal missing-value runs may contain merged "
            f"cells: {merged_cell_columns!r}."
        )

    return warnings


def validate_table(df: pd.DataFrame) -> dict[str, Any]:
    """Return a structured, read-only validation report for ``df``.

    Null values and whitespace-only strings are counted as missing for
    reporting purposes.  The input DataFrame, its labels, dtypes, and values
    are never changed.
    """

    _ensure_dataframe(df)

    dimensions = get_dataframe_dimensions(df)
    missing_cells = count_missing_cells(df)
    missing_by_column = count_missing_by_column(df)
    empty_rows = find_empty_rows(df)
    empty_columns = find_empty_columns(df)
    duplicate_row_positions = find_duplicate_rows(df)
    non_empty_by_row = count_non_empty_cells_by_row(df)
    title_rows = detect_title_like_rows(df)
    header_rows = detect_possible_header_rows(
        df,
        title_rows=title_rows,
    )
    date_cells = find_date_like_cells(df)
    hierarchical_header = detect_possible_hierarchical_header(
        df,
        header_rows=header_rows,
    )
    merged_cell_columns = detect_merged_cell_like_columns(df)
    warnings = _build_warnings(
        missing_cells=missing_cells,
        duplicate_rows=len(duplicate_row_positions),
        empty_rows=empty_rows,
        empty_columns=empty_columns,
        title_rows=title_rows,
        header_rows=header_rows,
        date_cell_count=len(date_cells),
        hierarchical_header=hierarchical_header,
        merged_cell_columns=merged_cell_columns,
    )

    return {
        "rows": dimensions["rows"],
        "columns": dimensions["columns"],
        "missing_cells": missing_cells,
        "missing_values_by_column": missing_by_column,
        "duplicate_rows": len(duplicate_row_positions),
        "duplicate_row_positions": duplicate_row_positions,
        "empty_rows": empty_rows,
        "empty_columns": empty_columns,
        "non_empty_cells_per_row": non_empty_by_row,
        "possible_title_rows": title_rows,
        "possible_header_rows": header_rows,
        "date_like_cells": len(date_cells),
        "date_like_cell_locations": date_cells,
        "possible_hierarchical_header": hierarchical_header,
        "possible_merged_cell_columns": merged_cell_columns,
        "warnings": warnings,
    }

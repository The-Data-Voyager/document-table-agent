"""Profile-specific transformations for IDSP weekly outbreak reports."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


_UNIQUE_ID_PATTERN = re.compile(r"^[A-Z]{2}/[A-Z]{3}/\d{4}/\d+/\d+$")
_CURRENT_COLUMNS = (
    "Unique_ID",
    "State_UT",
    "District",
    "Disease_Illness",
    "Cases",
    "Deaths",
    "Outbreak_Start_Date",
    "Reporting_Date",
    "Current_Status",
    "Comments_Action_Taken",
)
_LATE_COLUMNS = (
    "Unique_ID",
    "State_UT",
    "District",
    "Disease_Illness",
    "Cases",
    "Deaths",
    "Outbreak_Start_Date",
    "Current_Status",
    "Comments_Action_Taken",
)
_DATE_FORMAT = "%d-%m-%Y"


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


def _clean_text(value: Any) -> str:
    if _is_missing(value):
        return ""
    return " ".join(str(value).split())


def _clean_wrapped_name(value: Any) -> str:
    """Join a final one-letter line caused by a narrow name column."""

    if _is_missing(value):
        return ""
    parts = [" ".join(line.split()) for line in str(value).splitlines()]
    parts = [part for part in parts if part]
    if (
        len(parts) >= 2
        and len(parts[-1]) == 1
        and parts[-1].isalpha()
        and parts[-2][-1:].isalpha()
    ):
        parts[-2] = parts[-2] + parts[-1]
        parts.pop()
    return " ".join(parts)


def _record_ranges(df: pd.DataFrame) -> list[tuple[int, int]]:
    starts = [
        position
        for position in range(len(df))
        if _UNIQUE_ID_PATTERN.fullmatch(
            _clean_text(df.iloc[position, 0])
        )
    ]
    if not starts:
        raise ValueError("No outbreak Unique ID rows were found.")
    return [
        (start, starts[index + 1] if index + 1 < len(starts) else len(df))
        for index, start in enumerate(starts)
    ]


def _joined_comments(record: pd.DataFrame, start_column: int) -> str:
    parts = []
    for row_position in range(len(record)):
        for column_position in range(start_column, record.shape[1]):
            value = _clean_text(record.iloc[row_position, column_position])
            if value:
                parts.append(value)
    return " ".join(parts)


def _prepared_table(
    records: list[list[Any]],
    columns: tuple[str, ...],
) -> pd.DataFrame:
    if not records:
        raise ValueError("No logical outbreak records were reconstructed.")
    unique_ids = [record[0] for record in records]
    if len(unique_ids) != len(set(unique_ids)):
        raise ValueError("Reconstructed outbreak records contain duplicate IDs.")
    return pd.DataFrame([list(columns), *records])


def prepare_current_outbreak_table(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse continued physical lines into current-week outbreak rows."""

    _ensure_dataframe(df)
    if df.shape[1] < 12:
        raise ValueError("Current outbreak extraction must have at least 12 columns.")
    original = df.copy(deep=True)
    records = []
    for start, end in _record_ranges(df):
        record = df.iloc[start:end]
        first = record.iloc[0]
        values = [
            (
                _clean_wrapped_name(first.iloc[position])
                if position in {1, 2}
                else _clean_text(first.iloc[position])
            )
            for position in range(9)
        ]
        values.append(_joined_comments(record, 9))
        records.append(values)
    result = _prepared_table(records, _CURRENT_COLUMNS)
    pd.testing.assert_frame_equal(df, original, check_exact=True)
    return result


def prepare_late_outbreak_table(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse late-report rows despite the continuation page's column shift."""

    _ensure_dataframe(df)
    if df.shape[1] < 11:
        raise ValueError("Late outbreak extraction must have at least 11 columns.")
    original = df.copy(deep=True)
    records = []
    for start, end in _record_ranges(df):
        record = df.iloc[start:end]
        first = record.iloc[0]
        shifted_layout = _is_missing(first.iloc[1]) and not _is_missing(
            first.iloc[2]
        )
        if shifted_layout:
            positions = (0, 2, 3, 4, 5, 6, 7, 10)
            comment_start = 11
        else:
            positions = (0, 1, 2, 3, 4, 5, 6, 7)
            comment_start = 8
        values = [
            (
                _clean_wrapped_name(first.iloc[position])
                if output_position in {1, 2}
                else _clean_text(first.iloc[position])
            )
            for output_position, position in enumerate(positions)
        ]
        values.append(_joined_comments(record, comment_start))
        records.append(values)
    result = _prepared_table(records, _LATE_COLUMNS)
    pd.testing.assert_frame_equal(df, original, check_exact=True)
    return result


def _to_analysis(
    df: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    date_columns: tuple[str, ...],
) -> pd.DataFrame:
    _ensure_dataframe(df)
    if tuple(df.columns) != columns:
        raise ValueError(
            f"Outbreak table columns do not match the expected schema: {columns!r}."
        )
    original = df.copy(deep=True)
    result = df.copy(deep=True)
    for column in date_columns:
        try:
            result[column] = pd.to_datetime(
                result[column],
                format=_DATE_FORMAT,
                errors="raise",
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Column {column!r} must contain DD-MM-YYYY dates."
            ) from error
    for column in ("Cases", "Deaths"):
        result[column] = pd.to_numeric(result[column], errors="raise")
    pd.testing.assert_frame_equal(df, original, check_exact=True)
    return result


def current_outbreaks_to_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates and numeric measures for current-week outbreak records."""

    return _to_analysis(
        df,
        columns=_CURRENT_COLUMNS,
        date_columns=("Outbreak_Start_Date", "Reporting_Date"),
    )


def late_outbreaks_to_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates and numeric measures for late-reported outbreak records."""

    return _to_analysis(
        df,
        columns=_LATE_COLUMNS,
        date_columns=("Outbreak_Start_Date",),
    )

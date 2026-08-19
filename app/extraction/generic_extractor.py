"""Guided, profile-free table discovery for text-based PDF documents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pdfplumber

from app.extraction.table_extractor import _validate_pdf_path, table_to_dataframe


_DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097f]+")
_EMPTY_PARENTHESES_PATTERN = re.compile(r"\(\s*\)")
_NUMBER_PATTERN = re.compile(
    r"^[+-]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)%?$"
)
_DATE_PATTERN = re.compile(
    r"^(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|"
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2})$"
)


@dataclass
class GenericTableCandidate:
    """One table detected on one human-numbered PDF page."""

    page_number: int
    table_index: int
    dataframe: pd.DataFrame

    @property
    def label(self) -> str:
        rows, columns = self.dataframe.shape
        return (
            f"Page {self.page_number} - Table {self.table_index + 1} "
            f"({rows} rows x {columns} columns)"
        )


def discover_table_candidates(
    pdf_path: str | Path,
    page_numbers: tuple[int, ...] | list[int],
) -> tuple[GenericTableCandidate, ...]:
    """Return every table pdfplumber detects on the requested pages."""

    path = _validate_pdf_path(pdf_path)
    pages = tuple(page_numbers)
    if not pages:
        raise ValueError("At least one page number is required.")
    if any(isinstance(page, bool) or not isinstance(page, int) for page in pages):
        raise TypeError("Page numbers must contain only integers.")
    if pages != tuple(sorted(set(pages))):
        raise ValueError("Page numbers must be ascending and unique.")

    candidates: list[GenericTableCandidate] = []
    with pdfplumber.open(path) as pdf:
        total_pages = len(pdf.pages)
        if pages[0] < 1 or pages[-1] > total_pages:
            raise ValueError(
                f"Page numbers must be between 1 and {total_pages}."
            )
        for page_number in pages:
            for table_index, table in enumerate(
                pdf.pages[page_number - 1].extract_tables()
            ):
                if table:
                    candidates.append(
                        GenericTableCandidate(
                            page_number=page_number,
                            table_index=table_index,
                            dataframe=table_to_dataframe(table),
                        )
                    )
    return tuple(candidates)


def _clean_text(value: str, *, remove_devanagari: bool) -> str:
    if remove_devanagari:
        value = _DEVANAGARI_PATTERN.sub("", value)
        value = _EMPTY_PARENTHESES_PATTERN.sub(" ", value)
    return " ".join(value.split())


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _is_data_like(value: object) -> bool:
    if _is_missing(value):
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return True
    if not isinstance(value, str):
        return False
    text = value.strip()
    return bool(_NUMBER_PATTERN.fullmatch(text) or _DATE_PATTERN.fullmatch(text))


def suggest_header_rows(dataframe: pd.DataFrame) -> tuple[int, ...]:
    """Suggest consecutive header rows immediately before dense table data."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    if dataframe.empty or dataframe.shape[1] == 0:
        return ()

    scan_limit = min(15, len(dataframe))
    data_threshold = max(2, int(dataframe.shape[1] * 0.4))
    data_start = None
    for row_position in range(scan_limit):
        values = dataframe.iloc[row_position].tolist()
        if sum(_is_data_like(value) for value in values) >= data_threshold:
            data_start = row_position
            break
    if data_start is None or data_start == 0:
        return ()

    header_rows: list[int] = []
    sparse_threshold = max(1, dataframe.shape[1] // 5)
    for row_position in range(data_start - 1, -1, -1):
        populated = sum(
            not _is_missing(value) and str(value).strip() != ""
            for value in dataframe.iloc[row_position].tolist()
        )
        if populated <= sparse_threshold or len(header_rows) == 4:
            break
        header_rows.append(row_position)
    return tuple(sorted(header_rows))


def _combined_headers(header_block: pd.DataFrame) -> list[str]:
    grouped = header_block.ffill(axis=1)
    headers: list[str] = []
    final_counts: dict[str, int] = {}

    for column_position in range(grouped.shape[1]):
        parts: list[str] = []
        seen: set[str] = set()
        for value in grouped.iloc[:, column_position].tolist():
            if _is_missing(value):
                continue
            part = str(value).strip()
            normalized = part.casefold()
            if part and normalized not in seen:
                parts.append(part)
                seen.add(normalized)
        base = " | ".join(parts) or f"Column_{column_position + 1}"
        final_counts[base] = final_counts.get(base, 0) + 1
        suffix = f"_{final_counts[base]}" if final_counts[base] > 1 else ""
        headers.append(f"{base}{suffix}")
    return headers


def clean_generic_table(
    dataframe: pd.DataFrame,
    *,
    header_row: int | None = None,
    header_row_count: int = 1,
    drop_empty: bool = True,
    remove_devanagari: bool = False,
) -> pd.DataFrame:
    """Apply only user-selected, layout-agnostic cleanup to a raw table."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    if header_row is not None and (
        isinstance(header_row, bool)
        or not isinstance(header_row, int)
        or header_row < 0
        or header_row >= len(dataframe)
    ):
        raise ValueError("header_row must identify an existing row position.")
    if (
        isinstance(header_row_count, bool)
        or not isinstance(header_row_count, int)
        or header_row_count < 1
    ):
        raise ValueError("header_row_count must be a positive integer.")
    if header_row is None and header_row_count != 1:
        raise ValueError("header_row_count requires a selected header_row.")
    if header_row is not None and header_row + header_row_count > len(dataframe):
        raise ValueError("The selected header rows extend beyond the table.")
    if not isinstance(remove_devanagari, bool):
        raise TypeError("remove_devanagari must be a boolean.")

    cleaned = dataframe.copy(deep=True)
    cleaned = cleaned.map(
        lambda value: (
            _clean_text(value, remove_devanagari=remove_devanagari)
            if isinstance(value, str)
            else value
        )
    )

    if header_row is not None:
        header_end = header_row + header_row_count
        cleaned.columns = _combined_headers(cleaned.iloc[header_row:header_end])
        cleaned = cleaned.iloc[header_end:].copy()

    if drop_empty:
        empty = cleaned.isna() | cleaned.map(
            lambda value: isinstance(value, str) and not value.strip()
        )
        cleaned = cleaned.loc[~empty.all(axis=1)]
        empty = cleaned.isna() | cleaned.map(
            lambda value: isinstance(value, str) and not value.strip()
        )
        cleaned = cleaned.loc[:, ~empty.all(axis=0)]

    return cleaned.reset_index(drop=True)

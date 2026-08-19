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
        sections = split_table_sections(self.dataframe)
        section_text = (
            f", {len(sections)} logical sections" if len(sections) > 1 else ""
        )
        return (
            f"Page {self.page_number} - Table {self.table_index + 1} "
            f"({rows} rows x {columns} columns{section_text})"
        )


@dataclass
class GenericTableSection:
    """One logical table inside a larger PDF grid extraction."""

    start_row: int
    end_row: int
    title: str
    dataframe: pd.DataFrame

    @property
    def label(self) -> str:
        rows, columns = self.dataframe.shape
        return f"{self.title} ({rows} rows x {columns} columns)"


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


def _row_text(dataframe: pd.DataFrame, row_position: int) -> str:
    parts = [
        str(value)
        for value in dataframe.iloc[row_position].tolist()
        if not _is_missing(value) and str(value).strip()
    ]
    return _clean_text(" ".join(parts), remove_devanagari=True)


def split_table_sections(
    dataframe: pd.DataFrame,
) -> tuple[GenericTableSection, ...]:
    """Split vertically stacked numbered tables detected as one PDF grid."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    if dataframe.empty:
        return ()

    starts = [0]
    maximum_title_cells = max(2, dataframe.shape[1] // 4)
    for row_position in range(1, len(dataframe)):
        values = dataframe.iloc[row_position].tolist()
        populated = sum(
            not _is_missing(value) and str(value).strip() != ""
            for value in values
        )
        text = _row_text(dataframe, row_position)
        if (
            populated <= maximum_title_cells
            and re.search(r"(?:^|\s)\d+\.\s+[A-Za-z]", text)
        ):
            starts.append(row_position)

    sections: list[GenericTableSection] = []
    boundaries = [*starts, len(dataframe)]
    for section_index, (start, end) in enumerate(
        zip(boundaries, boundaries[1:]),
        start=1,
    ):
        text = _row_text(dataframe, start)
        title_match = re.search(r"(?:^|\s)(\d+\.\s+.+)", text)
        title = (
            title_match.group(1).strip()
            if title_match
            else f"Table section {section_index}"
        )
        if len(title) > 90:
            title = f"{title[:87].rstrip()}..."
        sections.append(
            GenericTableSection(
                start_row=start,
                end_row=end,
                title=title,
                dataframe=dataframe.iloc[start:end].copy().reset_index(drop=True),
            )
        )
    return tuple(sections)


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
    trailing_sparse_rows: list[int] = []
    found_dense_header = False
    sparse_threshold = max(1, dataframe.shape[1] // 5)
    for row_position in range(data_start - 1, -1, -1):
        populated = sum(
            not _is_missing(value) and str(value).strip() != ""
            for value in dataframe.iloc[row_position].tolist()
        )
        if len(header_rows) + len(trailing_sparse_rows) == 4:
            break
        if populated <= sparse_threshold:
            if found_dense_header:
                break
            trailing_sparse_rows.append(row_position)
            continue
        header_rows.append(row_position)
        if not found_dense_header:
            header_rows.extend(trailing_sparse_rows)
            trailing_sparse_rows.clear()
            found_dense_header = True
    return tuple(sorted(header_rows))


def _combined_headers(header_block: pd.DataFrame) -> list[str]:
    grouped = header_block.copy(deep=True)
    for row_position in range(len(grouped)):
        populated = sum(
            not _is_missing(value) and str(value).strip() != ""
            for value in grouped.iloc[row_position].tolist()
        )
        if populated > 1:
            grouped.iloc[row_position] = grouped.iloc[row_position].ffill()
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


def _compact_split_header_columns(
    dataframe: pd.DataFrame,
    *,
    header_row: int,
    header_row_count: int,
) -> pd.DataFrame:
    """Move detached header text onto adjacent unlabeled data columns."""

    header_end = header_row + header_row_count
    header = dataframe.iloc[header_row:header_end]
    data = dataframe.iloc[header_end:]

    def has_content(series: pd.Series) -> bool:
        return any(
            not _is_missing(value) and str(value).strip() != ""
            for value in series.tolist()
        )

    header_only = [
        column
        for column in dataframe.columns
        if has_content(header[column]) and not has_content(data[column])
    ]
    unlabeled_data = [
        column
        for column in dataframe.columns
        if not has_content(header[column]) and has_content(data[column])
    ]
    if not header_only or not unlabeled_data:
        return dataframe

    compacted = dataframe.copy(deep=True)
    removed = []
    column_positions = {
        column: position for position, column in enumerate(dataframe.columns)
    }
    for source in header_only:
        destination = min(
            unlabeled_data,
            key=lambda column: (
                abs(column_positions[column] - column_positions[source]),
                column_positions[column] > column_positions[source],
                column_positions[column],
            ),
        )
        for row_position in range(header_row, header_end):
            value = compacted.at[row_position, source]
            if _is_missing(value) or not str(value).strip():
                continue
            existing = compacted.at[row_position, destination]
            compacted.at[row_position, destination] = (
                value
                if _is_missing(existing) or not str(existing).strip()
                else f"{existing}\n{value}"
            )
        removed.append(source)

    return compacted.drop(columns=removed)


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
        cleaned = _compact_split_header_columns(
            cleaned,
            header_row=header_row,
            header_row_count=header_row_count,
        )
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

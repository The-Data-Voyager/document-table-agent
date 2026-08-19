"""Guided, profile-free table discovery for text-based PDF documents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pdfplumber

from app.extraction.table_extractor import _validate_pdf_path, table_to_dataframe


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


def clean_generic_table(
    dataframe: pd.DataFrame,
    *,
    header_row: int | None = None,
    drop_empty: bool = True,
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

    cleaned = dataframe.copy(deep=True)
    cleaned = cleaned.map(
        lambda value: value.strip() if isinstance(value, str) else value
    )

    if header_row is not None:
        raw_headers = cleaned.iloc[header_row].tolist()
        headers: list[str] = []
        counts: dict[str, int] = {}
        for position, value in enumerate(raw_headers, start=1):
            base = str(value).strip() if value is not None else ""
            base = base or f"Column_{position}"
            counts[base] = counts.get(base, 0) + 1
            suffix = f"_{counts[base]}" if counts[base] > 1 else ""
            headers.append(f"{base}{suffix}")
        cleaned = cleaned.iloc[header_row + 1 :].copy()
        cleaned.columns = headers

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

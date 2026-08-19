"""Application services shared by the CLI and local web interface."""

from __future__ import annotations

import io
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

from app.agent import NaturalLanguageQueryResult, NaturalLanguageTableAgent
from app.agent.builtin_semantics import BUILTIN_SEMANTIC_CATALOGS
from app.extraction.generic_extractor import (
    GenericTableCandidate,
    discover_table_candidates,
)
from app.extraction.table_extractor import extract_table_span_as_dataframe
from app.parsers.pdf_parser import get_page_count, pdf_has_text, search_pdf
from app.pipeline.builtin_profiles import BUILTIN_DOCUMENT_PROFILES
from app.pipeline.profiles import DocumentProfile
from app.pipeline.runner import DocumentPipelineResult, run_profiled_pipeline


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_GUIDED_PAGES = 30


class InvalidPdfUploadError(ValueError):
    """Raised when uploaded content is not a supported PDF payload."""


@dataclass(frozen=True)
class PdfInspection:
    """Basic facts needed to configure guided extraction controls."""

    page_count: int
    has_extractable_text: bool


@dataclass(frozen=True)
class GenericDiscoveryResult:
    """Tables discovered without selecting a document profile."""

    page_count: int
    searched_pages: tuple[int, ...]
    candidates: tuple[GenericTableCandidate, ...]
    keyword: str | None = None


def validate_pdf_upload(
    pdf_bytes: bytes,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> bytes:
    """Validate an uploaded PDF without trusting its client-side filename."""

    if not isinstance(pdf_bytes, bytes):
        raise TypeError("pdf_bytes must be bytes.")
    if not pdf_bytes:
        raise InvalidPdfUploadError("The uploaded file is empty.")
    if len(pdf_bytes) > max_bytes:
        size_mb = len(pdf_bytes) / (1024 * 1024)
        limit_mb = max_bytes / (1024 * 1024)
        raise InvalidPdfUploadError(
            f"The PDF is {size_mb:.1f} MB; the interface limit is "
            f"{limit_mb:.0f} MB."
        )
    if b"%PDF-" not in pdf_bytes[:1024]:
        raise InvalidPdfUploadError(
            "The uploaded content does not have a valid PDF header."
        )
    return pdf_bytes


@contextmanager
def _temporary_pdf_path(pdf_bytes: bytes) -> Iterator[Path]:
    validated = validate_pdf_upload(pdf_bytes)
    with tempfile.TemporaryDirectory(prefix="document-table-agent-") as temp_dir:
        pdf_path = Path(temp_dir) / "uploaded.pdf"
        pdf_path.write_bytes(validated)
        yield pdf_path


def process_pdf_bytes(
    pdf_bytes: bytes,
    *,
    profiles: Sequence[DocumentProfile] = BUILTIN_DOCUMENT_PROFILES,
) -> DocumentPipelineResult:
    """Process PDF bytes using automatic, profile-based document detection."""

    with _temporary_pdf_path(pdf_bytes) as pdf_path:
        return run_profiled_pipeline(pdf_path, tuple(profiles))


def inspect_pdf_bytes(pdf_bytes: bytes) -> PdfInspection:
    """Return page count and whether a PDF contains searchable text."""

    with _temporary_pdf_path(pdf_bytes) as pdf_path:
        return PdfInspection(
            page_count=get_page_count(pdf_path),
            has_extractable_text=pdf_has_text(pdf_path),
        )


def _validated_page_range(
    start_page: int,
    end_page: int,
    *,
    total_pages: int,
) -> tuple[int, ...]:
    if any(
        isinstance(page, bool) or not isinstance(page, int)
        for page in (start_page, end_page)
    ):
        raise TypeError("Page numbers must be integers.")
    if start_page < 1 or end_page > total_pages or start_page > end_page:
        raise ValueError(
            f"Choose an ascending page range between 1 and {total_pages}."
        )
    pages = tuple(range(start_page, end_page + 1))
    if len(pages) > MAX_GUIDED_PAGES:
        raise ValueError(
            f"Guided extraction is limited to {MAX_GUIDED_PAGES} pages at a "
            "time. Choose a smaller range."
        )
    return pages


def discover_pdf_tables_by_page(
    pdf_bytes: bytes,
    *,
    start_page: int,
    end_page: int,
) -> GenericDiscoveryResult:
    """Discover every table in a user-selected inclusive page range."""

    with _temporary_pdf_path(pdf_bytes) as pdf_path:
        total_pages = get_page_count(pdf_path)
        pages = _validated_page_range(
            start_page,
            end_page,
            total_pages=total_pages,
        )
        return GenericDiscoveryResult(
            page_count=total_pages,
            searched_pages=pages,
            candidates=discover_table_candidates(pdf_path, pages),
        )


def discover_pdf_tables_by_keyword(
    pdf_bytes: bytes,
    keyword: str,
) -> GenericDiscoveryResult:
    """Find pages containing a title/keyword and discover their tables."""

    normalized_keyword = keyword.strip()
    if not normalized_keyword:
        raise ValueError("Enter a non-empty table name or keyword.")
    with _temporary_pdf_path(pdf_bytes) as pdf_path:
        total_pages = get_page_count(pdf_path)
        pages = tuple(search_pdf(pdf_path, normalized_keyword))
        if len(pages) > MAX_GUIDED_PAGES:
            raise ValueError(
                f"The keyword matched more than {MAX_GUIDED_PAGES} pages. "
                "Use a more specific table title."
            )
        return GenericDiscoveryResult(
            page_count=total_pages,
            searched_pages=pages,
            candidates=discover_table_candidates(pdf_path, pages) if pages else (),
            keyword=normalized_keyword,
        )


def extract_pdf_table_span(
    pdf_bytes: bytes,
    *,
    start_page: int,
    end_page: int,
    table_index: int,
) -> pd.DataFrame:
    """Extract the same detected table number across consecutive pages."""

    with _temporary_pdf_path(pdf_bytes) as pdf_path:
        total_pages = get_page_count(pdf_path)
        pages = _validated_page_range(
            start_page,
            end_page,
            total_pages=total_pages,
        )
        return extract_table_span_as_dataframe(
            pdf_path,
            pages,
            table_index=table_index,
        )


def analysis_tables(
    result: DocumentPipelineResult,
) -> dict[str, pd.DataFrame]:
    """Build stable query names for postprocessed pipeline tables."""

    tables: dict[str, pd.DataFrame] = {}
    for table_name, table_result in result.tables.items():
        derived = table_result.postprocessed_tables
        if len(derived) == 1:
            tables[table_name] = next(iter(derived.values()))
        else:
            for output_name, table in derived.items():
                tables[f"{table_name}.{output_name}"] = table
    if not tables:
        raise ValueError(
            "The selected profile produced no analysis-ready tables to query."
        )
    return tables


def ask_document_question(
    result: DocumentPipelineResult,
    question: str,
) -> NaturalLanguageQueryResult:
    """Ask a supported English question about a processed document."""

    catalog = BUILTIN_SEMANTIC_CATALOGS.get(result.profile_name)
    if catalog is None:
        raise ValueError(
            "No English-question catalog is registered for profile "
            f"{result.profile_name!r}."
        )
    return NaturalLanguageTableAgent(analysis_tables(result), catalog).ask(
        question
    )


def _profile_by_name(
    profile_name: str,
    profiles: Sequence[DocumentProfile],
) -> DocumentProfile:
    matches = [profile for profile in profiles if profile.name == profile_name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one profile named {profile_name!r}; found {len(matches)}."
        )
    return matches[0]


def downloadable_tables(
    result: DocumentPipelineResult,
    *,
    profiles: Sequence[DocumentProfile] = BUILTIN_DOCUMENT_PROFILES,
) -> dict[str, pd.DataFrame]:
    """Return configured CSV filenames paired with their in-memory tables."""

    profile = _profile_by_name(result.profile_name, profiles)
    table_profiles = {table.name: table for table in profile.tables}
    downloads: dict[str, pd.DataFrame] = {}

    for table_name, table_result in result.tables.items():
        table_profile = table_profiles[table_name]
        clean_filename = (
            table_profile.output_filename or f"{table_name}_clean.csv"
        )
        downloads[clean_filename] = table_result.transformed_table

        postprocessors = {
            item.name: item for item in table_profile.postprocessors
        }
        for output_name, table in table_result.postprocessed_tables.items():
            postprocessor = postprocessors[output_name]
            filename = (
                postprocessor.output_filename
                or f"{table_name}_{output_name}.csv"
            )
            if filename in downloads:
                raise ValueError(f"Duplicate download filename: {filename!r}.")
            downloads[filename] = table
    return downloads


def dataframe_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a table as an Excel-friendly UTF-8 CSV payload."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    return df.to_csv(index=False).encode("utf-8-sig")


def build_download_zip(tables: Mapping[str, pd.DataFrame]) -> bytes:
    """Bundle named DataFrames as CSV files in an in-memory ZIP archive."""

    if not tables:
        raise ValueError("At least one table is required to build a ZIP file.")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, table in tables.items():
            safe_name = Path(filename).name
            if safe_name != filename or not safe_name.lower().endswith(".csv"):
                raise ValueError(f"Invalid CSV filename: {filename!r}.")
            archive.writestr(safe_name, dataframe_csv_bytes(table))
    return buffer.getvalue()

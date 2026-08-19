"""Application services shared by the CLI and local web interface."""

from __future__ import annotations

import io
import re
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
    clean_generic_table,
    discover_table_candidates,
    find_suspicious_columns,
    split_table_sections,
    suggest_header_rows,
)
from app.extraction.table_extractor import extract_table_span_as_dataframe
from app.parsers.pdf_parser import get_page_count, pdf_has_text, search_pdf
from app.pipeline.builtin_profiles import BUILTIN_DOCUMENT_PROFILES
from app.pipeline.profiles import DocumentProfile
from app.pipeline.runner import DocumentPipelineResult, run_profiled_pipeline


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_GUIDED_PAGES = 30
MAX_AUTOMATIC_PAGES = 100
MAX_BATCH_FILES = 10
MAX_BATCH_BYTES = 75 * 1024 * 1024


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

    @property
    def logical_table_count(self) -> int:
        """Count logical sections after splitting stacked PDF grids."""

        return sum(
            max(1, len(split_table_sections(candidate.dataframe)))
            for candidate in self.candidates
        )

    @property
    def pages_without_candidates(self) -> tuple[int, ...]:
        """Return searched pages where pdfplumber found no table grid."""

        detected_pages = {candidate.page_number for candidate in self.candidates}
        return tuple(
            page for page in self.searched_pages if page not in detected_pages
        )


@dataclass(frozen=True)
class BatchDocumentResult:
    """Extraction outcome for one uploaded document in a batch."""

    filename: str
    method: str
    tables: Mapping[str, pd.DataFrame]
    warnings: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class BatchProcessingResult:
    """Ordered collection of per-document batch outcomes."""

    documents: tuple[BatchDocumentResult, ...]

    @property
    def successful_documents(self) -> int:
        return sum(item.error is None for item in self.documents)

    @property
    def table_count(self) -> int:
        return sum(len(item.tables) for item in self.documents)


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


def discover_pdf_tables_automatically(
    pdf_bytes: bytes,
) -> GenericDiscoveryResult:
    """Discover table grids across every page of a reasonably sized PDF."""

    with _temporary_pdf_path(pdf_bytes) as pdf_path:
        total_pages = get_page_count(pdf_path)
        if total_pages > MAX_AUTOMATIC_PAGES:
            raise ValueError(
                f"Automatic whole-document discovery is limited to "
                f"{MAX_AUTOMATIC_PAGES} pages. Use Page or page range for "
                "larger documents."
            )
        pages = tuple(range(1, total_pages + 1))
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
            normalize_column_layout=True,
        )


def _automatically_clean_discovery(
    discovery: GenericDiscoveryResult,
) -> tuple[dict[str, pd.DataFrame], tuple[str, ...]]:
    tables: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []
    for candidate in discovery.candidates:
        sections = split_table_sections(candidate.dataframe)
        for section_index, section in enumerate(sections, start=1):
            raw_table = section.dataframe
            suggested = suggest_header_rows(raw_table)
            cleaned = clean_generic_table(
                raw_table,
                header_row=suggested[0] if suggested else None,
                header_row_count=len(suggested) if suggested else 1,
                remove_devanagari=True,
                remove_repeated_headers=True,
            )
            if cleaned.empty or cleaned.shape[1] == 0:
                continue
            name = (
                f"page_{candidate.page_number}_table_"
                f"{candidate.table_index + 1}"
            )
            if len(sections) > 1:
                name = f"{name}_section_{section_index}"
            filename = f"{name}_clean.csv"
            tables[filename] = cleaned
            suspicious = find_suspicious_columns(cleaned)
            if suspicious:
                warnings.append(
                    f"{filename}: review possible clipped/split columns "
                    f"{list(suspicious)!r}."
                )
    return tables, tuple(warnings)


def process_pdf_batch(
    documents: Sequence[tuple[str, bytes]],
) -> BatchProcessingResult:
    """Process several PDFs using profiles first and generic discovery second."""

    items = tuple(documents)
    if not items:
        raise ValueError("Upload at least one PDF for batch processing.")
    if len(items) > MAX_BATCH_FILES:
        raise ValueError(
            f"Batch processing is limited to {MAX_BATCH_FILES} PDFs at once."
        )
    total_bytes = sum(len(payload) for _, payload in items)
    if total_bytes > MAX_BATCH_BYTES:
        raise ValueError(
            "The combined batch exceeds the 75 MB processing limit."
        )

    results: list[BatchDocumentResult] = []
    for filename, payload in items:
        try:
            validate_pdf_upload(payload)
            try:
                profile_result = process_pdf_bytes(payload)
            except Exception:
                discovery = discover_pdf_tables_automatically(payload)
                tables, warnings = _automatically_clean_discovery(discovery)
                if not tables:
                    raise ValueError(
                        "No extractable text-table grids were found; OCR may "
                        "be required."
                    )
                results.append(
                    BatchDocumentResult(
                        filename=filename,
                        method="Automatic pdfplumber discovery",
                        tables=tables,
                        warnings=warnings,
                    )
                )
            else:
                results.append(
                    BatchDocumentResult(
                        filename=filename,
                        method=f"Profile: {profile_result.profile_name}",
                        tables=downloadable_tables(profile_result),
                    )
                )
        except Exception as error:
            results.append(
                BatchDocumentResult(
                    filename=filename,
                    method="Failed",
                    tables={},
                    error=str(error),
                )
            )
    return BatchProcessingResult(tuple(results))


def batch_downloadable_tables(
    result: BatchProcessingResult,
) -> dict[str, pd.DataFrame]:
    """Flatten successful batch tables into safe, unique ZIP filenames."""

    if not isinstance(result, BatchProcessingResult):
        raise TypeError("result must be a BatchProcessingResult.")
    flattened: dict[str, pd.DataFrame] = {}
    for document_index, document in enumerate(result.documents, start=1):
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(document.filename).stem)
        stem = stem.strip("_") or f"document_{document_index}"
        for filename, table in document.tables.items():
            candidate = f"{stem}__{Path(filename).name}"
            suffix = 2
            while candidate in flattened:
                candidate = f"{stem}_{suffix}__{Path(filename).name}"
                suffix += 1
            flattened[candidate] = table
    if not flattened:
        raise ValueError("The batch produced no tables to download.")
    return flattened


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

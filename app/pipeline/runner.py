"""Reusable orchestration for configured document/table profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.export.csv_exporter import export_dataframe_to_csv
from app.extraction.table_extractor import (
    extract_table_as_dataframe,
    extract_table_span_as_dataframe,
)
from app.parsers.pdf_parser import search_pdf
from app.pipeline.profiles import (
    DocumentProfile,
    OutputSchemaProfile,
    TableProfile,
    TableTransformationProfile,
)
from app.transformation.transformer import (
    assign_values_from_mapping,
    transform_table,
)
from app.validation.table_validator import validate_table


class UnknownDocumentProfileError(ValueError):
    """Raised when no configured document profile matches."""


class AmbiguousDocumentProfileError(ValueError):
    """Raised when more than one configured document profile matches."""


@dataclass
class TablePipelineResult:
    """Artifacts and observations produced for one table profile."""

    profile_name: str
    page_number: int
    page_numbers: tuple[int, ...]
    raw_table: pd.DataFrame
    validation_report: dict[str, Any]
    transformed_table: pd.DataFrame
    postprocessed_tables: dict[str, pd.DataFrame]
    warnings: list[str]
    output_path: Path | None = None
    postprocessed_output_paths: dict[str, Path] | None = None


@dataclass
class DocumentPipelineResult:
    """Results produced by one selected document profile."""

    profile_name: str
    tables: dict[str, TablePipelineResult]


def _profile_matches(
    pdf_path: str | Path,
    profile: DocumentProfile,
    term_cache: dict[str, list[int]],
) -> bool:
    for term in profile.detection_terms:
        if term not in term_cache:
            term_cache[term] = search_pdf(pdf_path, term)
        if not term_cache[term]:
            return False
    return True


def detect_document_profile(
    pdf_path: str | Path,
    profiles: tuple[DocumentProfile, ...] | list[DocumentProfile],
) -> DocumentProfile:
    """Return the only profile whose detection terms occur in the PDF.

    Detection never guesses: zero matches and multiple matches are distinct
    errors that require a new marker or an explicit profile choice.
    """

    available_profiles = tuple(profiles)
    if not available_profiles:
        raise ValueError("At least one document profile is required.")
    if any(
        not isinstance(profile, DocumentProfile)
        for profile in available_profiles
    ):
        raise TypeError("profiles must contain only DocumentProfile instances.")

    term_cache: dict[str, list[int]] = {}
    matches = [
        profile
        for profile in available_profiles
        if _profile_matches(pdf_path, profile, term_cache)
    ]
    if not matches:
        raise UnknownDocumentProfileError(
            "No configured document profile matched the PDF. Add a profile "
            "or select one explicitly after reviewing the layout."
        )
    if len(matches) > 1:
        raise AmbiguousDocumentProfileError(
            "Multiple document profiles matched the PDF: "
            f"{[profile.name for profile in matches]!r}."
        )
    return matches[0]


def _locate_table_page(
    pdf_path: str | Path,
    table_profile: TableProfile,
) -> int:
    page_sets = [
        set(search_pdf(pdf_path, term))
        for term in table_profile.search_terms
    ]
    matching_pages = sorted(set.intersection(*page_sets))
    if not matching_pages:
        raise ValueError(
            f"No page contains all search terms for table profile "
            f"{table_profile.name!r}."
        )
    if table_profile.page_match_index >= len(matching_pages):
        raise IndexError(
            f"page_match_index {table_profile.page_match_index} is not "
            f"available for matching pages {matching_pages}."
        )
    return matching_pages[table_profile.page_match_index]


def _locate_table_pages(
    pdf_path: str | Path,
    table_profile: TableProfile,
) -> tuple[int, ...]:
    """Resolve a single page or a marker-bounded inclusive page sequence."""

    start_page = _locate_table_page(pdf_path, table_profile)
    if not table_profile.page_end_search_terms:
        return (start_page,)

    end_page_sets = [
        set(search_pdf(pdf_path, term))
        for term in table_profile.page_end_search_terms
    ]
    end_pages = sorted(set.intersection(*end_page_sets))
    candidates = [page for page in end_pages if page > start_page]
    if not candidates:
        raise ValueError(
            f"No end-marker page after page {start_page} for table profile "
            f"{table_profile.name!r}."
        )
    marker_page = candidates[0]
    last_page = (
        marker_page if table_profile.include_end_page else marker_page - 1
    )
    if last_page < start_page:
        raise ValueError(
            f"The resolved page span is empty for {table_profile.name!r}."
        )
    return tuple(range(start_page, last_page + 1))


def transform_dataframe_with_profile(
    df: pd.DataFrame,
    profile: TableTransformationProfile,
) -> pd.DataFrame:
    """Transform a DataFrame using only explicit profile configuration."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not isinstance(profile, TableTransformationProfile):
        raise TypeError("profile must be a TableTransformationProfile.")

    original = df.copy(deep=True)
    working = df
    if profile.preprocessor is not None:
        working = profile.preprocessor(df)
        if not isinstance(working, pd.DataFrame):
            raise TypeError("Profile preprocessor must return a DataFrame.")
        pd.testing.assert_frame_equal(df, original, check_exact=True)

    configured_columns = (
        profile.identity_column_positions
        + profile.measure_column_positions
        + profile.forward_fill_column_positions
    )
    if profile.mapping is not None:
        configured_columns += (
            profile.mapping.key_column_position,
            profile.mapping.target_column_position,
        )
    out_of_range = sorted(
        {
            position
            for position in configured_columns
            if position >= working.shape[1]
        }
    )
    if out_of_range:
        raise IndexError(
            "Profile column positions are outside the DataFrame: "
            f"{out_of_range}."
        )

    transformed = transform_table(
        working,
        header_row_positions=profile.header_row_positions,
        fill_merged_headers_from_column=(
            profile.fill_merged_headers_from_column
        ),
        forward_fill_column_positions=(
            profile.forward_fill_column_positions
        ),
        numeric_column_positions=profile.measure_column_positions,
        remove_devanagari=profile.remove_devanagari,
        normalize_text_whitespace=profile.normalize_text_whitespace,
    )
    if profile.mapping is not None:
        transformed = assign_values_from_mapping(
            transformed,
            key_column_position=profile.mapping.key_column_position,
            target_column_position=profile.mapping.target_column_position,
            mapping=profile.mapping.values,
            strict=profile.mapping.strict,
        )

    pd.testing.assert_frame_equal(df, original, check_exact=True)
    return transformed


def _validate_output_schema(
    df: pd.DataFrame,
    schema: OutputSchemaProfile,
    *,
    context: str,
) -> None:
    if df.columns.nlevels != schema.column_levels:
        raise ValueError(
            f"{context} has {df.columns.nlevels} column levels; expected "
            f"{schema.column_levels}."
        )
    if schema.column_count is not None and df.shape[1] != schema.column_count:
        raise ValueError(
            f"{context} has {df.shape[1]} columns; expected "
            f"{schema.column_count}."
        )
    if (
        schema.expected_row_count is not None
        and df.shape[0] != schema.expected_row_count
    ):
        raise ValueError(
            f"{context} has {df.shape[0]} rows; expected "
            f"{schema.expected_row_count}."
        )

    required = set(schema.required_columns)
    missing_required = [
        column for column in required if column not in df.columns
    ]
    if missing_required:
        raise ValueError(
            f"{context} is missing required columns: {missing_required!r}."
        )

    for column in schema.non_null_columns:
        if column not in df.columns:
            raise ValueError(
                f"{context} cannot check missing values because column "
                f"{column!r} is absent."
            )
        series = df[column]
        missing_mask = series.isna()
        if pd.api.types.is_object_dtype(series.dtype) or isinstance(
            series.dtype,
            pd.StringDtype,
        ):
            missing_mask = missing_mask | series.map(
                lambda value: isinstance(value, str) and not value.strip()
            )
        if bool(missing_mask.any()):
            rows = [
                int(position)
                for position, missing in enumerate(missing_mask.tolist())
                if bool(missing)
            ]
            raise ValueError(
                f"{context} has missing values in {column!r} at row "
                f"positions {rows}."
            )

    if schema.unique_key_columns:
        missing_keys = [
            column
            for column in schema.unique_key_columns
            if column not in df.columns
        ]
        if missing_keys:
            raise ValueError(
                f"{context} cannot check uniqueness; columns are absent: "
                f"{missing_keys!r}."
            )
        duplicate_mask = df.duplicated(
            subset=list(schema.unique_key_columns),
            keep=False,
        )
        if bool(duplicate_mask.any()):
            rows = [
                int(position)
                for position, duplicate in enumerate(duplicate_mask.tolist())
                if bool(duplicate)
            ]
            raise ValueError(
                f"{context} has duplicate key rows at positions {rows}."
            )


def _run_table_profile(
    pdf_path: str | Path,
    table_profile: TableProfile,
    *,
    output_dir: str | Path | None,
    overwrite: bool,
) -> TablePipelineResult:
    page_numbers = _locate_table_pages(pdf_path, table_profile)
    page_number = page_numbers[0]
    if len(page_numbers) == 1:
        raw_table = extract_table_as_dataframe(
            pdf_path=pdf_path,
            page_number=page_number,
            table_index=table_profile.table_index,
        )
    else:
        raw_table = extract_table_span_as_dataframe(
            pdf_path=pdf_path,
            page_numbers=page_numbers,
            table_index=table_profile.table_index,
        )
    validation_report = validate_table(raw_table)
    configured_headers = list(
        table_profile.transformation.header_row_positions
    )
    detected_headers = validation_report["possible_header_rows"]
    warnings = []
    if (
        table_profile.transformation.preprocessor is None
        and configured_headers != detected_headers
    ):
        warnings.append(
            "Configured header rows differ from validation hints: "
            f"configured={configured_headers}, detected={detected_headers}. "
            "The explicit profile was used."
        )

    transformed_table = transform_dataframe_with_profile(
        raw_table,
        table_profile.transformation,
    )
    wide_schema = table_profile.output_schema or OutputSchemaProfile(
        column_levels=len(
            table_profile.transformation.header_row_positions
        )
    )
    _validate_output_schema(
        transformed_table,
        wide_schema,
        context=f"Transformed table {table_profile.name!r}",
    )

    output_path = None
    if output_dir is not None and table_profile.output_filename is not None:
        output_path = export_dataframe_to_csv(
            transformed_table,
            Path(output_dir) / table_profile.output_filename,
            overwrite=overwrite,
        )

    postprocessed_tables = {}
    postprocessed_output_paths = {}
    for postprocessor in table_profile.postprocessors:
        transformed_snapshot = transformed_table.copy(deep=True)
        postprocessed = postprocessor.processor(transformed_table)
        if not isinstance(postprocessed, pd.DataFrame):
            raise TypeError(
                f"Postprocessor {postprocessor.name!r} must return a "
                "pandas DataFrame."
            )
        pd.testing.assert_frame_equal(
            transformed_table,
            transformed_snapshot,
            check_exact=True,
        )
        _validate_output_schema(
            postprocessed,
            postprocessor.output_schema,
            context=(
                f"Postprocessed table {table_profile.name!r}/"
                f"{postprocessor.name!r}"
            ),
        )
        postprocessed_tables[postprocessor.name] = postprocessed
        if output_dir is not None and postprocessor.output_filename is not None:
            postprocessed_output_paths[postprocessor.name] = (
                export_dataframe_to_csv(
                    postprocessed,
                    Path(output_dir) / postprocessor.output_filename,
                    overwrite=overwrite,
                )
            )

    return TablePipelineResult(
        profile_name=table_profile.name,
        page_number=page_number,
        page_numbers=page_numbers,
        raw_table=raw_table,
        validation_report=validation_report,
        transformed_table=transformed_table,
        postprocessed_tables=postprocessed_tables,
        warnings=warnings,
        output_path=output_path,
        postprocessed_output_paths=postprocessed_output_paths,
    )


def run_document_profile(
    pdf_path: str | Path,
    profile: DocumentProfile,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> DocumentPipelineResult:
    """Run every table configured in one explicit document profile."""

    if not isinstance(profile, DocumentProfile):
        raise TypeError("profile must be a DocumentProfile.")
    results = {
        table_profile.name: _run_table_profile(
            pdf_path,
            table_profile,
            output_dir=output_dir,
            overwrite=overwrite,
        )
        for table_profile in profile.tables
    }
    return DocumentPipelineResult(profile_name=profile.name, tables=results)


def run_profiled_pipeline(
    pdf_path: str | Path,
    profiles: tuple[DocumentProfile, ...] | list[DocumentProfile],
    *,
    profile_name: str | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> DocumentPipelineResult:
    """Detect or explicitly select a profile, then run its configured tables."""

    available_profiles = tuple(profiles)
    if profile_name is None:
        selected_profile = detect_document_profile(
            pdf_path,
            available_profiles,
        )
    else:
        named_matches = [
            profile
            for profile in available_profiles
            if profile.name == profile_name
        ]
        if not named_matches:
            raise UnknownDocumentProfileError(
                f"Unknown document profile: {profile_name!r}."
            )
        if len(named_matches) > 1:
            raise AmbiguousDocumentProfileError(
                f"Duplicate document profile name: {profile_name!r}."
            )
        selected_profile = named_matches[0]

    return run_document_profile(
        pdf_path,
        selected_profile,
        output_dir=output_dir,
        overwrite=overwrite,
    )

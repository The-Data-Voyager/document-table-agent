import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from app.document_service import (
    InvalidPdfUploadError,
    analysis_tables,
    batch_downloadable_tables,
    build_download_zip,
    dataframe_csv_bytes,
    downloadable_tables,
    process_pdf_batch,
    process_pdf_bytes,
    validate_pdf_upload,
)
from app.pipeline.runner import DocumentPipelineResult, TablePipelineResult


def _table_result(
    *,
    name: str,
    clean: pd.DataFrame,
    derived_name: str,
    derived: pd.DataFrame,
) -> TablePipelineResult:
    return TablePipelineResult(
        profile_name=name,
        page_number=1,
        page_numbers=(1,),
        raw_table=clean.copy(),
        validation_report={},
        transformed_table=clean,
        postprocessed_tables={derived_name: derived},
        warnings=[],
    )


def test_validate_pdf_upload_rejects_empty_oversized_and_non_pdf_content():
    with pytest.raises(InvalidPdfUploadError, match="empty"):
        validate_pdf_upload(b"")
    with pytest.raises(InvalidPdfUploadError, match="interface limit"):
        validate_pdf_upload(b"%PDF-1.7 extra", max_bytes=5)
    with pytest.raises(InvalidPdfUploadError, match="valid PDF header"):
        validate_pdf_upload(b"plain text")


def test_process_pdf_bytes_detects_the_electricity_profile():
    sample_pdf = next(
        path
        for path in Path("sample_documents").glob("*.pdf")
        if path.name.startswith("Weekly ")
    )

    result = process_pdf_bytes(sample_pdf.read_bytes())

    assert result.profile_name == "grid_india_weekly_report"
    assert set(result.tables) == {"energy_consumption", "maximum_demand"}
    assert analysis_tables(result)["energy_consumption"].shape == (280, 4)


def test_batch_processing_uses_profiles_and_builds_prefixed_downloads():
    sample_pdf = next(
        path
        for path in Path("sample_documents").glob("*.pdf")
        if path.name.startswith("Weekly ")
    )

    payload = sample_pdf.read_bytes()
    result = process_pdf_batch(
        (
            ("weekly report.pdf", payload),
            ("weekly report copy.pdf", payload),
        )
    )
    downloads = batch_downloadable_tables(result)

    assert result.successful_documents == 2
    assert result.table_count == 8
    assert all(
        document.method.startswith("Profile:")
        for document in result.documents
    )
    assert any(name.startswith("weekly_report__") for name in downloads)
    assert any(name.startswith("weekly_report_copy__") for name in downloads)


def test_downloadable_tables_use_profile_configured_filenames():
    clean = pd.DataFrame({"State": ["A"]})
    long = pd.DataFrame(
        {
            "Region": ["R"],
            "State": ["A"],
            "Date": [pd.Timestamp("2026-01-01")],
            "Energy_Consumption_MU": [1.5],
        }
    )
    result = DocumentPipelineResult(
        profile_name="grid_india_weekly_report",
        tables={
            "energy_consumption": _table_result(
                name="energy_consumption",
                clean=clean,
                derived_name="long",
                derived=long,
            )
        },
    )

    downloads = downloadable_tables(result)

    assert set(downloads) == {
        "clean_energy_consumption.csv",
        "energy_consumption_long.csv",
    }
    assert dataframe_csv_bytes(long).startswith(b"\xef\xbb\xbf")


def test_build_download_zip_contains_csv_payloads():
    tables = {
        "first.csv": pd.DataFrame({"A": [1]}),
        "second.csv": pd.DataFrame({"B": [2]}),
    }

    payload = build_download_zip(tables)

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert set(archive.namelist()) == set(tables)
        assert archive.read("first.csv").startswith(b"\xef\xbb\xbfA")


def test_build_download_zip_rejects_unsafe_filename():
    with pytest.raises(ValueError, match="Invalid CSV filename"):
        build_download_zip({"../table.csv": pd.DataFrame({"A": [1]})})

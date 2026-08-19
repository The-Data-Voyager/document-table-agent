from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from app.pipeline import runner
from app.pipeline.builtin_profiles import (
    BUILTIN_DOCUMENT_PROFILES,
    GRID_INDIA_WEEKLY_REPORT,
)
from app.pipeline.outbreak_profiles import IDSP_WEEKLY_OUTBREAK_REPORT
from app.pipeline.profiles import (
    ColumnMappingProfile,
    DocumentProfile,
    TableProfile,
    TableTransformationProfile,
)
from app.pipeline.runner import (
    AmbiguousDocumentProfileError,
    UnknownDocumentProfileError,
    detect_document_profile,
    run_document_profile,
    transform_dataframe_with_profile,
)


def _table_profile(name: str, search_term: str) -> TableProfile:
    return TableProfile(
        name=name,
        search_terms=(search_term,),
        transformation=TableTransformationProfile(
            header_row_positions=(0,),
            identity_column_positions=(0,),
            measure_column_positions=(1,),
        ),
    )


def test_generic_profile_transforms_non_electricity_sales_table():
    raw = pd.DataFrame(
        [
            ["Quarterly sales", None, None],
            ["Product", "Units", "Revenue"],
            ["Books", "10", "125.50"],
            ["Games", "4", "80.00"],
        ]
    )
    original = raw.copy(deep=True)
    profile = TableTransformationProfile(
        header_row_positions=(1,),
        identity_column_positions=(0,),
        measure_column_positions=(1, 2),
    )

    transformed = transform_dataframe_with_profile(raw, profile)

    assert transformed.columns.tolist() == ["Product", "Units", "Revenue"]
    assert transformed["Units"].tolist() == [10, 4]
    assert transformed["Revenue"].tolist() == [125.5, 80.0]
    assert_frame_equal(raw, original, check_exact=True)


def test_generic_profile_supports_mapping_without_region_logic():
    raw = pd.DataFrame(
        [
            ["Team", "Owner", "Score"],
            [None, "Alice", "10"],
            [None, "Bob", "20"],
        ]
    )
    profile = TableTransformationProfile(
        header_row_positions=(0,),
        identity_column_positions=(0, 1),
        measure_column_positions=(2,),
        mapping=ColumnMappingProfile(
            key_column_position=1,
            target_column_position=0,
            values={"Alice": "Blue", "Bob": "Green"},
        ),
    )

    transformed = transform_dataframe_with_profile(raw, profile)

    assert transformed["Team"].tolist() == ["Blue", "Green"]
    assert transformed["Score"].tolist() == [10, 20]


def test_profile_rejects_overlapping_identity_and_measure_columns():
    with pytest.raises(ValueError, match="cannot overlap"):
        TableTransformationProfile(
            header_row_positions=(0,),
            identity_column_positions=(0, 1),
            measure_column_positions=(1, 2),
        )


def test_transform_rejects_out_of_range_identity_column():
    raw = pd.DataFrame([["Name", "Value"], ["A", "1"]])
    profile = TableTransformationProfile(
        header_row_positions=(0,),
        identity_column_positions=(5,),
        measure_column_positions=(1,),
    )

    with pytest.raises(IndexError, match="outside"):
        transform_dataframe_with_profile(raw, profile)


def test_detect_document_profile_requires_one_unambiguous_match(monkeypatch):
    alpha = DocumentProfile(
        name="alpha_document",
        detection_terms=("alpha marker",),
        tables=(_table_profile("alpha_table", "alpha table"),),
    )
    beta = DocumentProfile(
        name="beta_document",
        detection_terms=("beta marker",),
        tables=(_table_profile("beta_table", "beta table"),),
    )
    pages = {"alpha marker": [1], "beta marker": []}
    monkeypatch.setattr(
        runner,
        "search_pdf",
        lambda _path, term: pages[term],
    )

    assert detect_document_profile("sample.pdf", [alpha, beta]) is alpha


def test_detect_document_profile_rejects_unknown_and_ambiguous(monkeypatch):
    alpha = DocumentProfile(
        name="alpha_document",
        detection_terms=("alpha",),
        tables=(_table_profile("alpha_table", "alpha table"),),
    )
    beta = DocumentProfile(
        name="beta_document",
        detection_terms=("beta",),
        tables=(_table_profile("beta_table", "beta table"),),
    )

    monkeypatch.setattr(runner, "search_pdf", lambda _path, _term: [])
    with pytest.raises(UnknownDocumentProfileError):
        detect_document_profile("sample.pdf", [alpha, beta])

    monkeypatch.setattr(runner, "search_pdf", lambda _path, _term: [1])
    with pytest.raises(AmbiguousDocumentProfileError):
        detect_document_profile("sample.pdf", [alpha, beta])


def test_builtin_profile_runs_sample_pdf_end_to_end_without_page_constants():
    pdf_path = next(
        path
        for path in Path("sample_documents").glob("*.pdf")
        if path.name.startswith("Weekly ")
    )

    result = run_document_profile(pdf_path, GRID_INDIA_WEEKLY_REPORT)

    energy = result.tables["energy_consumption"].transformed_table
    maximum = result.tables["maximum_demand"].transformed_table
    energy_long = result.tables["energy_consumption"].postprocessed_tables[
        "long"
    ]
    maximum_long = result.tables["maximum_demand"].postprocessed_tables[
        "long"
    ]
    assert result.profile_name == "grid_india_weekly_report"
    assert energy.shape == (40, 9)
    assert maximum.shape == (39, 16)
    assert energy.iloc[:, 0].isna().sum() == 0
    assert maximum.iloc[:, 0].isna().sum() == 0
    assert maximum.columns.nlevels == 2
    assert energy_long.shape == (280, 4)
    assert maximum_long.shape == (273, 5)
    assert not energy_long.duplicated(
        subset=["Region", "State", "Date"]
    ).any()
    assert not maximum_long.duplicated(
        subset=["Region", "State", "Date"]
    ).any()
    assert BUILTIN_DOCUMENT_PROFILES == (
        GRID_INDIA_WEEKLY_REPORT,
        IDSP_WEEKLY_OUTBREAK_REPORT,
    )


def test_second_builtin_profile_stitches_outbreak_tables_across_pages():
    pdf_path = next(
        path
        for path in Path("sample_documents").glob("*.pdf")
        if path.name == "75788759701752062509.pdf"
    )

    detected = detect_document_profile(pdf_path, BUILTIN_DOCUMENT_PROFILES)
    result = run_document_profile(pdf_path, detected)

    current_result = result.tables["current_outbreaks"]
    late_result = result.tables["late_outbreaks"]
    current = current_result.postprocessed_tables["analysis"]
    late = late_result.postprocessed_tables["analysis"]
    assert detected is IDSP_WEEKLY_OUTBREAK_REPORT
    assert result.profile_name == "idsp_weekly_outbreak_report"
    assert current_result.page_numbers == tuple(range(3, 15))
    assert late_result.page_numbers == (15, 16)
    assert current.shape == (34, 10)
    assert late.shape == (5, 9)
    assert current["Unique_ID"].nunique() == 34
    assert late["Unique_ID"].nunique() == 5
    assert set(current["Unique_ID"]).isdisjoint(set(late["Unique_ID"]))
    assert pd.api.types.is_datetime64_any_dtype(
        current["Outbreak_Start_Date"]
    )
    assert pd.api.types.is_datetime64_any_dtype(current["Reporting_Date"])
    assert pd.api.types.is_datetime64_any_dtype(late["Outbreak_Start_Date"])
    assert "Maharashtra" in set(current["State_UT"])
    assert "Hepatitis A" in set(late["Disease_Illness"])

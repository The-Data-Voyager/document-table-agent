import pandas as pd
from pandas.testing import assert_frame_equal

from app.pipeline.profiles import TableTransformationProfile
from app.pipeline.runner import transform_dataframe_with_profile
from app.transformation.outbreak_analysis import (
    current_outbreaks_to_analysis,
    late_outbreaks_to_analysis,
    prepare_current_outbreak_table,
    prepare_late_outbreak_table,
)


def _padded(rows, width):
    return pd.DataFrame([row + [None] * (width - len(row)) for row in rows])


def test_current_preprocessor_collapses_comment_lines_and_wrapped_names():
    raw = _padded(
        [
            ["header"],
            [
                "MH/ABC/2025/19/001",
                "Maharashtr\na",
                "Pune",
                "Hepatitis\nA",
                "10",
                "1",
                "01-05-2025",
                "02-05-2025",
                "Under\nControl",
                "",
                "First line",
                "",
            ],
            [None] * 10 + ["second line", None],
            [
                "KL/XYZ/2025/19/002",
                "Kerala",
                "Ernakulam",
                "Dengue",
                "5",
                "0",
                "03-05-2025",
                "04-05-2025",
                "Under Surveillance",
                "",
                "Another record",
                "",
            ],
        ],
        12,
    )
    original = raw.copy(deep=True)
    profile = TableTransformationProfile(
        header_row_positions=(0,),
        identity_column_positions=(0, 1, 2, 3, 6, 7, 8, 9),
        measure_column_positions=(4, 5),
        preprocessor=prepare_current_outbreak_table,
    )

    transformed = transform_dataframe_with_profile(raw, profile)

    assert transformed.shape == (2, 10)
    assert transformed.loc[0, "State_UT"] == "Maharashtra"
    assert transformed.loc[0, "Disease_Illness"] == "Hepatitis A"
    assert transformed.loc[0, "Comments_Action_Taken"] == (
        "First line second line"
    )
    assert transformed["Cases"].tolist() == [10, 5]
    assert_frame_equal(raw, original, check_exact=True)


def test_late_preprocessor_handles_both_page_column_layouts():
    raw = _padded(
        [
            [
                "AP/ABC/2025/19/003",
                None,
                "Andhra\nPradesh",
                "Sri Sathya Sai",
                "Acute Diarrheal Disease",
                "33",
                "0",
                "19-02-2025",
                None,
                None,
                "Under Control",
                "",
                "Page fifteen comment",
                "",
            ],
            [
                "MG/XYZ/2025/19/004",
                "Meghalaya",
                "South Garo\nHills",
                "Human Rabies",
                "1",
                "1",
                "06-04-2025",
                "Under Surveillance",
                "",
                "Page sixteen comment",
                "",
            ],
        ],
        14,
    )

    prepared = prepare_late_outbreak_table(raw)

    assert prepared.shape == (3, 9)
    assert prepared.iloc[1, 1] == "Andhra Pradesh"
    assert prepared.iloc[2, 1] == "Meghalaya"
    assert prepared.iloc[1, 8] == "Page fifteen comment"
    assert prepared.iloc[2, 8] == "Page sixteen comment"


def test_outbreak_analysis_parses_dates_and_preserves_clean_inputs():
    current = pd.DataFrame(
        [
            [
                "MH/ABC/2025/19/001",
                "Maharashtra",
                "Pune",
                "Dengue",
                10,
                1,
                "01-05-2025",
                "02-05-2025",
                "Under Control",
                "Investigated.",
            ]
        ],
        columns=(
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
        ),
    )
    late = current.drop(columns="Reporting_Date")
    current_original = current.copy(deep=True)
    late_original = late.copy(deep=True)

    current_analysis = current_outbreaks_to_analysis(current)
    late_analysis = late_outbreaks_to_analysis(late)

    assert pd.api.types.is_datetime64_any_dtype(
        current_analysis["Outbreak_Start_Date"]
    )
    assert pd.api.types.is_datetime64_any_dtype(
        current_analysis["Reporting_Date"]
    )
    assert pd.api.types.is_datetime64_any_dtype(
        late_analysis["Outbreak_Start_Date"]
    )
    assert_frame_equal(current, current_original, check_exact=True)
    assert_frame_equal(late, late_original, check_exact=True)

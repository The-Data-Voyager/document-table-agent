import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from app.validation.table_validator import (
    count_missing_by_column,
    detect_merged_cell_like_columns,
    detect_possible_header_rows,
    detect_possible_hierarchical_header,
    detect_title_like_rows,
    find_date_like_cells,
    validate_table,
)


def test_normal_table_has_expected_dimensions_and_no_quality_issues():
    table = pd.DataFrame(
        {
            "name": ["Alice", "Bob"],
            "score": [10, 20],
        }
    )

    report = validate_table(table)

    assert report["rows"] == 2
    assert report["columns"] == 2
    assert report["missing_cells"] == 0
    assert report["duplicate_rows"] == 0
    assert report["empty_rows"] == []
    assert report["empty_columns"] == []
    assert report["possible_title_rows"] == []
    assert report["possible_hierarchical_header"] is False


def test_missing_cells_include_nulls_and_blank_strings():
    table = pd.DataFrame(
        {
            "group": ["A", None, "  "],
            "value": [1, 2, None],
        }
    )

    report = validate_table(table)

    assert report["missing_cells"] == 3
    assert count_missing_by_column(table) == {
        "group": 2,
        "value": 1,
    }
    assert report["non_empty_cells_per_row"] == [2, 1, 0]


def test_duplicate_rows_are_counted_after_first_occurrence():
    table = pd.DataFrame(
        {
            "label": ["A", "A", "B", "A"],
            "value": [1, 1, 2, 1],
        }
    )

    report = validate_table(table)

    assert report["duplicate_rows"] == 2
    assert report["duplicate_row_positions"] == [1, 3]


def test_completely_empty_row_is_reported_by_position():
    table = pd.DataFrame(
        [
            ["A", 1],
            [None, ""],
            ["B", 2],
        ],
        columns=["label", "value"],
    )

    report = validate_table(table)

    assert report["empty_rows"] == [1]


def test_completely_empty_column_is_reported_by_label():
    table = pd.DataFrame(
        {
            "label": ["A", "B"],
            "empty": [None, "  "],
        }
    )

    report = validate_table(table)

    assert report["empty_columns"] == ["empty"]


def test_title_row_and_following_header_row_are_detected():
    table = pd.DataFrame(
        [
            ["Quarterly performance report", None, None],
            ["Region", "Sales", "Cost"],
            ["North", 10, 8],
            ["South", 12, 9],
        ]
    )

    assert detect_title_like_rows(table) == [0]
    assert detect_possible_header_rows(table) == [1]


def test_valid_date_pattern_in_header_is_detected_conservatively():
    table = pd.DataFrame(
        [
            ["Region", "31-03-2026"],
            ["North", 10],
        ]
    )

    matches = find_date_like_cells(table)
    report = validate_table(table)

    assert matches == [
        {"row": 0, "column": 1, "value": "31-03-2026"}
    ]
    assert report["date_like_cells"] == 1
    assert report["possible_header_rows"] == [0]


def test_invalid_date_shaped_text_is_not_treated_as_a_date():
    table = pd.DataFrame([["Region", "31-02-2026"], ["North", 10]])

    assert find_date_like_cells(table) == []


def test_multi_level_header_like_structure_is_reported_cautiously():
    table = pd.DataFrame(
        [
            [None, "30-03-2026", None, "31-03-2026", None],
            ["Region", "Maximum", "Shortage", "Maximum", "Shortage"],
            ["North", 100, 0, 110, 1],
            ["South", 90, 0, 95, 0],
        ]
    )

    header_rows = detect_possible_header_rows(table)

    assert header_rows == [0, 1]
    assert detect_possible_hierarchical_header(
        table,
        header_rows=header_rows,
    ) is True
    assert validate_table(table)["possible_hierarchical_header"] is True


def test_internal_missing_runs_are_flagged_as_merged_cell_like():
    table = pd.DataFrame(
        {
            "group": ["A", None, None, "B", None, "TOTAL"],
            "item": ["a1", "a2", "a3", "b1", "b2", "all"],
            "value": [1, 2, 3, 4, 5, 15],
        }
    )

    assert detect_merged_cell_like_columns(table) == ["group"]


def test_validation_does_not_mutate_input_dataframe():
    table = pd.DataFrame(
        [
            ["Report title", None, None],
            ["Region", "31-03-2026", "01-04-2026"],
            ["North", 10.0, 11.0],
            [None, 12.0, 13.0],
        ],
        columns=["category", "first", "second"],
        index=[10, 20, 30, 40],
    )
    original = table.copy(deep=True)

    validate_table(table)

    assert_frame_equal(table, original, check_exact=True)


def test_non_dataframe_input_is_rejected():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        validate_table([["not", "a", "dataframe"]])

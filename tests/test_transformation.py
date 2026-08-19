import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from app.transformation.transformer import (
    assign_values_from_mapping,
    convert_columns_to_numeric,
    find_unmapped_key_values,
    forward_fill_columns,
    promote_multirow_header,
    promote_single_header_row,
    remove_devanagari_text,
    transform_table,
)


def test_promote_single_header_preserves_source_labels_and_input():
    raw = pd.DataFrame(
        [
            ["Report title", None, None],
            ["Region", "State", "Value"],
            ["North", "A", "1.5"],
            [None, "B", "2.0"],
        ]
    )
    original = raw.copy(deep=True)

    transformed = promote_single_header_row(raw, 1)

    assert transformed.columns.tolist() == ["Region", "State", "Value"]
    assert transformed.shape == (2, 3)
    assert transformed.iloc[0].tolist() == ["North", "A", "1.5"]
    assert_frame_equal(raw, original, check_exact=True)


def test_single_header_rejects_missing_labels_instead_of_inventing_names():
    raw = pd.DataFrame([["Title", None], ["Name", None], ["A", 1]])

    with pytest.raises(ValueError, match="missing or blank"):
        promote_single_header_row(raw, 1)


def test_multirow_header_preserves_levels_and_fills_selected_merged_band():
    raw = pd.DataFrame(
        [
            ["Report title", None, None, None, None, None],
            ["Region", "Date", "30-03", None, "31-03", None],
            [None, "State", "Maximum", "Shortage", "Maximum", "Shortage"],
            ["North", "A", "100", "0", "110", "1"],
        ]
    )
    original = raw.copy(deep=True)

    transformed = promote_multirow_header(
        raw,
        [1, 2],
        fill_merged_from_column=2,
    )

    assert transformed.columns.tolist() == [
        ("Region", ""),
        ("Date", "State"),
        ("30-03", "Maximum"),
        ("30-03", "Shortage"),
        ("31-03", "Maximum"),
        ("31-03", "Shortage"),
    ]
    assert transformed.iloc[0].tolist() == [
        "North",
        "A",
        "100",
        "0",
        "110",
        "1",
    ]
    assert_frame_equal(raw, original, check_exact=True)


def test_multirow_header_requires_consecutive_rows():
    raw = pd.DataFrame([["a", "b"], ["c", "d"], ["e", "f"]])

    with pytest.raises(ValueError, match="consecutive"):
        promote_multirow_header(raw, [0, 2])


def test_forward_fill_is_explicit_and_non_mutating():
    table = pd.DataFrame(
        {
            "group": ["A", None, "", "B", None],
            "value": [1, 2, 3, 4, 5],
        }
    )
    original = table.copy(deep=True)

    transformed = forward_fill_columns(table, [0])

    assert transformed["group"].tolist() == ["A", "A", "A", "B", "B"]
    assert_frame_equal(table, original, check_exact=True)


def test_numeric_conversion_changes_only_selected_copy():
    table = pd.DataFrame(
        {
            "label": ["A", "B"],
            "value": ["1.5", "2.0"],
            "untouched": ["3", "4"],
        }
    )
    original = table.copy(deep=True)

    transformed = convert_columns_to_numeric(table, [1])

    assert transformed["value"].tolist() == [1.5, 2.0]
    assert transformed["untouched"].tolist() == ["3", "4"]
    assert_frame_equal(table, original, check_exact=True)


def test_numeric_conversion_raises_on_unexpected_text_by_default():
    table = pd.DataFrame({"value": ["1", "not numeric"]})

    with pytest.raises(ValueError):
        convert_columns_to_numeric(table, [0])


def test_transform_table_runs_explicit_pipeline_without_mutation():
    raw = pd.DataFrame(
        [
            ["Title", None, None],
            ["Group", "Item", "Value"],
            ["A", "a1", "1"],
            [None, "a2", "2"],
        ]
    )
    original = raw.copy(deep=True)

    transformed = transform_table(
        raw,
        header_row_positions=[1],
        forward_fill_column_positions=[0],
        numeric_column_positions=[2],
    )

    assert transformed["Group"].tolist() == ["A", "A"]
    assert transformed["Value"].tolist() == [1, 2]
    assert_frame_equal(raw, original, check_exact=True)


def test_remove_devanagari_cleans_cells_and_flat_headers_only_on_copy():
    table = pd.DataFrame(
        {
            "क्षेत्र Region": ["उ०क्षे०\nNR", "पुडुचेरी (पॉण्डिचेरी) Pondy"],
            "मांग Demand": [100, 200],
        }
    )
    original = table.copy(deep=True)

    transformed = remove_devanagari_text(table)

    assert transformed.columns.tolist() == ["Region", "Demand"]
    assert transformed.iloc[:, 0].tolist() == ["NR", "Pondy"]
    assert transformed.iloc[:, 1].tolist() == [100, 200]
    assert_frame_equal(table, original, check_exact=True)


def test_remove_devanagari_cleans_every_multiindex_level():
    table = pd.DataFrame([[100, 0]])
    table.columns = pd.MultiIndex.from_tuples(
        [
            ("30-03-2026", "मांग Max Demand"),
            ("30-03-2026", "कमी Shortage"),
        ]
    )

    transformed = remove_devanagari_text(table)

    assert transformed.columns.tolist() == [
        ("30-03-2026", "Max Demand"),
        ("30-03-2026", "Shortage"),
    ]
    assert transformed.iloc[0].tolist() == [100, 0]


def test_hindi_only_cell_becomes_empty_without_affecting_numbers():
    table = pd.DataFrame({"label": ["केवल हिन्दी", 123]})

    transformed = remove_devanagari_text(table)

    assert transformed.iloc[0, 0] == ""
    assert transformed.iloc[1, 0] == 123


def test_transform_table_can_produce_english_only_copy():
    raw = pd.DataFrame(
        [
            ["शीर्षक Title", None, None],
            ["क्षेत्र Region", "राज्य State", "मांग Demand"],
            ["उत्तर North", "पंजाब Punjab", "100"],
        ]
    )
    original = raw.copy(deep=True)

    transformed = transform_table(
        raw,
        header_row_positions=[1],
        numeric_column_positions=[2],
        remove_devanagari=True,
    )

    assert transformed.columns.tolist() == ["Region", "State", "Demand"]
    assert transformed.iloc[0].tolist() == ["North", "Punjab", 100]
    assert_frame_equal(raw, original, check_exact=True)


def test_removed_hindi_initials_do_not_leave_orphan_periods():
    table = pd.DataFrame({"entity": ["आर.आई.एल. RIL Jamnagar"]})

    transformed = remove_devanagari_text(table)

    assert transformed.iloc[0, 0] == "RIL Jamnagar"


def test_find_unmapped_key_values_returns_distinct_non_missing_values():
    table = pd.DataFrame({"state": ["Punjab", "Unknown", "Unknown", None]})

    assert find_unmapped_key_values(table, 0, {"Punjab": "NR"}) == [
        "Unknown"
    ]


def test_assign_values_from_mapping_preserves_total_and_input():
    table = pd.DataFrame(
        {
            "region": [None, None, "ALL INDIA"],
            "state": ["Punjab", "Gujarat", None],
            "value": [1, 2, 3],
        }
    )
    original = table.copy(deep=True)

    transformed = assign_values_from_mapping(
        table,
        key_column_position=1,
        target_column_position=0,
        mapping={"Punjab": "NR", "Gujarat": "WR"},
    )

    assert transformed["region"].tolist() == ["NR", "WR", "ALL INDIA"]
    assert_frame_equal(table, original, check_exact=True)


def test_assign_values_from_mapping_fails_loudly_for_unknown_key():
    table = pd.DataFrame({"region": [None], "state": ["Unknown"]})

    with pytest.raises(ValueError, match="Unknown"):
        assign_values_from_mapping(
            table,
            key_column_position=1,
            target_column_position=0,
            mapping={"Punjab": "NR"},
        )

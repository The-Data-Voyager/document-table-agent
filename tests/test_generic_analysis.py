import pandas as pd
import pytest

from app.agent import (
    apply_column_corrections,
    ask_generic_table_question,
    build_chart_data,
    generic_question_examples,
    preferred_category_column,
    prepare_table_for_analysis,
)


def _table():
    return pd.DataFrame(
        {
            "State": ["Assam", "Bihar", "Assam"],
            "Cases": ["10", "20", "5"],
            "Deaths": ["1", "2", "0"],
        }
    )


def test_generic_questions_infer_numeric_columns_and_grouping():
    prepared = prepare_table_for_analysis(_table())
    result = ask_generic_table_question(
        _table(),
        "Show top 2 State by Cases",
    )

    assert pd.api.types.is_integer_dtype(prepared["Cases"])
    assert generic_question_examples(_table())[0] == (
        "What is the average Cases?"
    )
    assert result.answer.to_dict(orient="records") == [
        {"State": "Bihar", "Total_Cases": 20},
        {"State": "Assam", "Total_Cases": 15},
    ]


def test_generic_questions_recognize_state_inside_verbose_pdf_header():
    table = pd.DataFrame(
        {
            "Unique ID.": ["A-1", "A-2", "B-1"],
            "Name of State/UT": ["Assam", "Assam", "Odisha"],
            "No. of Cases": [10, 5, 20],
        }
    )

    result = ask_generic_table_question(
        table,
        "Which state has the most number of cases, name the state?",
    )

    assert preferred_category_column(
        table,
        excluded_columns=("No. of Cases",),
    ) == "Name of State/UT"
    assert generic_question_examples(table)[1] == (
        "Show top 5 Name of State/UT by No. of Cases"
    )
    assert result.answer.to_dict(orient="records") == [
        {"Name of State/UT": "Odisha", "Total_No. of Cases": 20}
    ]


def test_manual_column_corrections_are_non_mutating_and_validated():
    source = _table()
    corrected = apply_column_corrections(
        source,
        dropped_columns=("Deaths",),
        renamed_columns={"Cases": "Reported Cases"},
    )

    assert source.columns.tolist() == ["State", "Cases", "Deaths"]
    assert corrected.columns.tolist() == ["State", "Reported Cases"]
    with pytest.raises(ValueError, match="unique"):
        apply_column_corrections(
            source,
            renamed_columns={"Cases": "State"},
        )


def test_chart_data_aggregates_a_numeric_measure_by_category():
    chart = build_chart_data(
        _table(),
        x_column="State",
        y_column="Cases",
        aggregation="sum",
    )

    assert chart.to_dict(orient="records") == [
        {"State": "Assam", "Cases": 15},
        {"State": "Bihar", "Cases": 20},
    ]

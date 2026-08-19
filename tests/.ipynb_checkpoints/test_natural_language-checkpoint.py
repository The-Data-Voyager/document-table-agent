import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from app.agent import (
    AmbiguousQuestionError,
    DimensionSemantic,
    MetricSemantic,
    NaturalLanguageTableAgent,
    QuestionInterpretationError,
    SemanticCatalog,
    UnsupportedQuestionError,
)
from app.agent.builtin_semantics import GRID_INDIA_QUERY_SEMANTICS


@pytest.fixture
def analysis_tables():
    dates = pd.to_datetime(["2026-03-30", "2026-03-31"])
    energy = pd.DataFrame(
        {
            "Region": ["NR", "NR", "NR", "NR", "ALL INDIA", "ALL INDIA"],
            "State": [
                "Punjab",
                "Punjab",
                "Haryana",
                "Haryana",
                "ALL INDIA",
                "ALL INDIA",
            ],
            "Date": list(dates) * 3,
            "Energy_Consumption_MU": [10.0, 12.0, 20.0, 25.0, 30.0, 37.0],
        }
    )
    demand = pd.DataFrame(
        {
            "Region": ["NR", "NR", "NR", "NR", "ER", "ER"],
            "State": ["Delhi", "Delhi", "Haryana", "Haryana", "Odisha", "Odisha"],
            "Date": list(dates) * 3,
            "Maximum_Demand_MW": [90, 95, 100, 120, 80, 85],
            "Peak_Shortage_MW": [2, 0, 0, 1, 10, 20],
        }
    )
    return {
        "energy_consumption": energy,
        "maximum_demand": demand,
    }


def test_highest_energy_state_uses_sum_and_excludes_reported_total(
    analysis_tables,
):
    agent = NaturalLanguageTableAgent(
        analysis_tables,
        GRID_INDIA_QUERY_SEMANTICS,
    )

    result = agent.ask("Which state had the highest energy consumption?")

    assert result.answer.to_dict("records") == [
        {"State": "Haryana", "Total_Energy_Consumption_MU": 45.0}
    ]
    assert result.interpretation.request.limit == 1
    assert result.interpretation.request.filters[-1].value == "ALL INDIA"
    assert any("Excluded reported total" in note for note in result.interpretation.notes)


def test_state_and_date_range_question_returns_matching_demand_rows(
    analysis_tables,
):
    agent = NaturalLanguageTableAgent(
        analysis_tables,
        GRID_INDIA_QUERY_SEMANTICS,
    )

    result = agent.ask(
        "Show Delhi's demand from 30-03-2026 to 31-03-2026"
    )

    assert result.answer["State"].tolist() == ["Delhi", "Delhi"]
    assert result.answer["Maximum_Demand_MW"].tolist() == [90, 95]
    assert result.interpretation.request.filters[1].operator == "between"
    assert result.interpretation.request.filters[1].value == (
        "2026-03-30",
        "2026-03-31",
    )


def test_average_peak_shortage_by_region_is_grouped_and_sorted(
    analysis_tables,
):
    agent = NaturalLanguageTableAgent(
        analysis_tables,
        GRID_INDIA_QUERY_SEMANTICS,
    )

    result = agent.ask("Calculate average peak shortage by region")

    assert result.answer["Region"].tolist() == ["ER", "NR"]
    assert result.answer["Average_Peak_Shortage_MW"].tolist() == [15.0, 0.75]


def test_compare_two_states_returns_unaggregated_daily_rows(analysis_tables):
    agent = NaturalLanguageTableAgent(
        analysis_tables,
        GRID_INDIA_QUERY_SEMANTICS,
    )

    result = agent.ask("Compare maximum demand for Delhi and Haryana")

    assert result.answer.shape == (4, 5)
    assert result.answer["State"].tolist() == [
        "Delhi",
        "Delhi",
        "Haryana",
        "Haryana",
    ]
    assert result.interpretation.request.aggregations == ()


def test_parser_is_catalog_driven_for_non_electricity_table():
    sales = pd.DataFrame(
        {
            "Product": ["Books", "Books", "Games", "Games"],
            "Revenue": [10.0, 15.0, 40.0, 25.0],
        }
    )
    catalog = SemanticCatalog(
        name="sales",
        metrics=(
            MetricSemantic(
                name="revenue",
                table="sales",
                column="Revenue",
                aliases=("revenue",),
                default_aggregation="sum",
                display_columns=("Product", "Revenue"),
            ),
        ),
        dimensions=(
            DimensionSemantic(
                column="Product",
                aliases=("product", "products"),
            ),
        ),
    )
    agent = NaturalLanguageTableAgent({"sales": sales}, catalog)

    result = agent.ask("Show the top 2 products by revenue")

    assert result.answer.to_dict("records") == [
        {"Product": "Games", "Total_Revenue": 65.0},
        {"Product": "Books", "Total_Revenue": 25.0},
    ]


@pytest.mark.parametrize(
    "question, error_type, message",
    [
        (
            "Show energy and maximum demand",
            AmbiguousQuestionError,
            "multiple metrics",
        ),
        (
            "Compare demand for Delhi",
            AmbiguousQuestionError,
            "at least two recognized",
        ),
        (
            "Show total average energy",
            AmbiguousQuestionError,
            "Conflicting aggregation",
        ),
        (
            "What was the rainfall?",
            UnsupportedQuestionError,
            "does not identify",
        ),
        (
            "Show demand from 2026-04-05 to 2026-03-30",
            QuestionInterpretationError,
            "starts after",
        ),
    ],
)
def test_ambiguous_or_unsupported_questions_fail_explicitly(
    analysis_tables,
    question,
    error_type,
    message,
):
    agent = NaturalLanguageTableAgent(
        analysis_tables,
        GRID_INDIA_QUERY_SEMANTICS,
    )

    with pytest.raises(error_type, match=message):
        agent.ask(question)


def test_language_agent_owns_immutable_table_snapshots(analysis_tables):
    energy_original = analysis_tables["energy_consumption"].copy(deep=True)
    agent = NaturalLanguageTableAgent(
        analysis_tables,
        GRID_INDIA_QUERY_SEMANTICS,
    )

    agent.ask("Which state had the highest energy consumption?")

    assert_frame_equal(
        analysis_tables["energy_consumption"],
        energy_original,
        check_exact=True,
    )

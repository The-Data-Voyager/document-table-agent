import json
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from app.agent import (
    NoMatchingRowsError,
    QueryExecutionError,
    TableQueryAgent,
    UnknownColumnError,
    UnknownTableError,
    parse_query_request,
)
from app.pipeline.builtin_profiles import GRID_INDIA_WEEKLY_REPORT
from app.pipeline.runner import run_document_profile


@pytest.fixture
def energy_table():
    return pd.DataFrame(
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
            "Date": pd.to_datetime(
                [
                    "2026-03-30",
                    "2026-03-31",
                    "2026-03-30",
                    "2026-03-31",
                    "2026-03-30",
                    "2026-03-31",
                ]
            ),
            "Energy_Consumption_MU": [10.0, 12.0, 20.0, 25.0, 30.0, 37.0],
        }
    )


def test_parse_query_request_accepts_mapping_and_json():
    payload = {
        "table": "energy",
        "filters": [
            {"column": "State", "operator": "in", "value": ["Punjab"]}
        ],
        "group_by": ["State"],
        "aggregations": [
            {"column": "Energy", "operation": "mean"}
        ],
        "order_by": [{"column": "mean_Energy", "direction": "desc"}],
        "limit": 5,
    }

    from_mapping = parse_query_request(payload)
    from_json = parse_query_request(json.dumps(payload))

    assert from_mapping == from_json
    assert from_mapping.filters[0].value == ("Punjab",)
    assert from_mapping.aggregations[0].alias == "mean_Energy"


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"table": "energy", "unexpected": True}, "unsupported fields"),
        (
            {
                "table": "energy",
                "filters": [
                    {"column": "State", "operator": "eval", "value": "x"}
                ],
            },
            "Unsupported filter operator",
        ),
        (
            {"table": "energy", "group_by": ["State"]},
            "requires at least one aggregation",
        ),
        (
            {"table": "energy", "limit": 1001},
            "limit must be between",
        ),
    ],
)
def test_parse_query_request_rejects_unsafe_or_inconsistent_input(
    payload,
    message,
):
    with pytest.raises((TypeError, ValueError), match=message):
        parse_query_request(payload)


def test_agent_executes_grouped_aggregation_with_sort_limit_and_evidence(
    energy_table,
):
    original = energy_table.copy(deep=True)
    agent = TableQueryAgent({"energy": energy_table})

    result = agent.ask(
        {
            "table": "energy",
            "filters": [
                {"column": "Region", "operator": "ne", "value": "ALL INDIA"}
            ],
            "group_by": ["State"],
            "aggregations": [
                {
                    "column": "Energy_Consumption_MU",
                    "operation": "sum",
                    "alias": "Total_Energy_MU",
                }
            ],
            "order_by": [
                {"column": "Total_Energy_MU", "direction": "desc"}
            ],
            "limit": 1,
        }
    )

    assert result.answer.to_dict("records") == [
        {"State": "Haryana", "Total_Energy_MU": 45.0}
    ]
    assert result.source_row_count == 6
    assert result.matched_row_count == 4
    assert result.returned_row_count == 1
    assert result.evidence.shape == (4, 4)
    assert_frame_equal(energy_table, original, check_exact=True)


def test_agent_filters_dates_and_projects_source_columns(energy_table):
    agent = TableQueryAgent({"energy": energy_table})

    result = agent.ask(
        {
            "table": "energy",
            "columns": ["State", "Date", "Energy_Consumption_MU"],
            "filters": [
                {
                    "column": "State",
                    "operator": "in",
                    "value": ["Punjab", "Haryana"],
                },
                {
                    "column": "Date",
                    "operator": "between",
                    "value": ["2026-03-31", "2026-03-31"],
                },
            ],
            "order_by": [{"column": "Energy_Consumption_MU", "direction": "desc"}],
        }
    )

    assert result.answer["State"].tolist() == ["Haryana", "Punjab"]
    assert result.answer["Energy_Consumption_MU"].tolist() == [25.0, 12.0]
    assert pd.api.types.is_datetime64_any_dtype(result.answer["Date"])


def test_agent_supports_contains_without_expression_evaluation(energy_table):
    agent = TableQueryAgent({"energy": energy_table})

    result = agent.ask(
        {
            "table": "energy",
            "columns": ["State"],
            "filters": [
                {"column": "State", "operator": "contains", "value": "jab"}
            ],
        }
    )

    assert result.answer["State"].tolist() == ["Punjab", "Punjab"]


def test_agent_rejects_unknown_names_and_empty_matches(energy_table):
    agent = TableQueryAgent({"energy": energy_table})

    with pytest.raises(UnknownTableError, match="Unknown table"):
        agent.ask({"table": "missing"})
    with pytest.raises(UnknownColumnError, match="Unknown source columns"):
        agent.ask({"table": "energy", "columns": ["Missing"]})
    with pytest.raises(NoMatchingRowsError, match="matched no source rows"):
        agent.ask(
            {
                "table": "energy",
                "filters": [
                    {"column": "State", "operator": "eq", "value": "Goa"}
                ],
            }
        )


def test_agent_rejects_numeric_aggregation_on_text(energy_table):
    agent = TableQueryAgent({"energy": energy_table})

    with pytest.raises(QueryExecutionError, match="requires a numeric column"):
        agent.ask(
            {
                "table": "energy",
                "aggregations": [
                    {"column": "State", "operation": "mean"}
                ],
            }
        )


def test_agent_owns_a_snapshot_of_registered_tables(energy_table):
    agent = TableQueryAgent({"energy": energy_table})
    energy_table.loc[:, "Energy_Consumption_MU"] = -1

    result = agent.ask(
        {
            "table": "energy",
            "aggregations": [
                {
                    "column": "Energy_Consumption_MU",
                    "operation": "max",
                    "alias": "Maximum",
                }
            ],
        }
    )

    assert result.answer.loc[0, "Maximum"] == 37.0


def test_sample_pdf_pipeline_tables_are_queryable_end_to_end():
    pdf_path = next(
        path
        for path in Path("sample_documents").glob("*.pdf")
        if path.name.startswith("Weekly ")
    )
    pipeline_result = run_document_profile(
        pdf_path,
        GRID_INDIA_WEEKLY_REPORT,
    )
    tables = {
        table_name: table_result.postprocessed_tables["long"]
        for table_name, table_result in pipeline_result.tables.items()
    }
    agent = TableQueryAgent(tables)

    result = agent.ask(
        {
            "table": "maximum_demand",
            "filters": [
                {"column": "Region", "operator": "ne", "value": "ALL INDIA"}
            ],
            "group_by": ["Region"],
            "aggregations": [
                {
                    "column": "Peak_Shortage_MW",
                    "operation": "max",
                    "alias": "Maximum_Peak_Shortage_MW",
                }
            ],
            "order_by": [
                {
                    "column": "Maximum_Peak_Shortage_MW",
                    "direction": "desc",
                }
            ],
        }
    )

    descriptions = agent.describe_tables()
    assert descriptions["energy_consumption"]["rows"] == 280
    assert descriptions["maximum_demand"]["rows"] == 273
    assert result.answer.columns.tolist() == [
        "Region",
        "Maximum_Peak_Shortage_MW",
    ]
    assert not result.answer.empty
    assert result.matched_row_count == 273

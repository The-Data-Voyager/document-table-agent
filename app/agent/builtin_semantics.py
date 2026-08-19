"""Semantic catalogs associated with built-in document profiles."""

from app.agent.natural_language import (
    DimensionSemantic,
    MetricSemantic,
    SemanticCatalog,
    TotalExclusion,
)


GRID_INDIA_QUERY_SEMANTICS = SemanticCatalog(
    name="grid_india_weekly_report",
    metrics=(
        MetricSemantic(
            name="energy consumption",
            table="energy_consumption",
            column="Energy_Consumption_MU",
            aliases=(
                "energy consumption",
                "energy consumed",
                "energy",
            ),
            default_aggregation="sum",
            display_columns=(
                "Region",
                "State",
                "Date",
                "Energy_Consumption_MU",
            ),
            date_column="Date",
        ),
        MetricSemantic(
            name="maximum demand",
            table="maximum_demand",
            column="Maximum_Demand_MW",
            aliases=("maximum demand", "max demand", "demand"),
            default_aggregation="max",
            display_columns=(
                "Region",
                "State",
                "Date",
                "Maximum_Demand_MW",
                "Peak_Shortage_MW",
            ),
            date_column="Date",
        ),
        MetricSemantic(
            name="peak shortage",
            table="maximum_demand",
            column="Peak_Shortage_MW",
            aliases=("peak shortage", "peak deficit", "shortage"),
            default_aggregation="max",
            display_columns=(
                "Region",
                "State",
                "Date",
                "Maximum_Demand_MW",
                "Peak_Shortage_MW",
            ),
            date_column="Date",
        ),
    ),
    dimensions=(
        DimensionSemantic(column="Region", aliases=("region", "regions")),
        DimensionSemantic(column="State", aliases=("state", "states")),
    ),
    total_exclusions=(
        TotalExclusion(
            table="energy_consumption",
            column="Region",
            value="ALL INDIA",
        ),
    ),
)


_CURRENT_OUTBREAK_COLUMNS = (
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
)
_LATE_OUTBREAK_COLUMNS = (
    "Unique_ID",
    "State_UT",
    "District",
    "Disease_Illness",
    "Cases",
    "Deaths",
    "Outbreak_Start_Date",
    "Current_Status",
    "Comments_Action_Taken",
)

IDSP_OUTBREAK_QUERY_SEMANTICS = SemanticCatalog(
    name="idsp_weekly_outbreak_report",
    metrics=(
        MetricSemantic(
            name="current outbreak cases",
            table="current_outbreaks",
            column="Cases",
            aliases=(
                "current outbreak cases",
                "outbreak cases",
                "reported cases",
                "cases",
            ),
            default_aggregation="sum",
            display_columns=_CURRENT_OUTBREAK_COLUMNS,
            date_column="Outbreak_Start_Date",
        ),
        MetricSemantic(
            name="current outbreak deaths",
            table="current_outbreaks",
            column="Deaths",
            aliases=(
                "current outbreak deaths",
                "outbreak deaths",
                "reported deaths",
                "deaths",
            ),
            default_aggregation="sum",
            display_columns=_CURRENT_OUTBREAK_COLUMNS,
            date_column="Outbreak_Start_Date",
        ),
        MetricSemantic(
            name="late outbreak cases",
            table="late_outbreaks",
            column="Cases",
            aliases=(
                "late outbreak cases",
                "late reported cases",
                "late cases",
            ),
            default_aggregation="sum",
            display_columns=_LATE_OUTBREAK_COLUMNS,
            date_column="Outbreak_Start_Date",
        ),
        MetricSemantic(
            name="late outbreak deaths",
            table="late_outbreaks",
            column="Deaths",
            aliases=(
                "late outbreak deaths",
                "late reported deaths",
                "late deaths",
            ),
            default_aggregation="sum",
            display_columns=_LATE_OUTBREAK_COLUMNS,
            date_column="Outbreak_Start_Date",
        ),
    ),
    dimensions=(
        DimensionSemantic(
            column="State_UT",
            aliases=("state", "states", "state ut", "states uts"),
        ),
        DimensionSemantic(
            column="District",
            aliases=("district", "districts"),
        ),
        DimensionSemantic(
            column="Disease_Illness",
            aliases=("disease", "diseases", "illness", "illnesses"),
        ),
        DimensionSemantic(
            column="Current_Status",
            aliases=("status", "statuses"),
        ),
    ),
)


BUILTIN_SEMANTIC_CATALOGS = {
    GRID_INDIA_QUERY_SEMANTICS.name: GRID_INDIA_QUERY_SEMANTICS,
    IDSP_OUTBREAK_QUERY_SEMANTICS.name: IDSP_OUTBREAK_QUERY_SEMANTICS,
}

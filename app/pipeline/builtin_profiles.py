"""Built-in document profiles supplied with this project."""

from app.pipeline.profiles import (
    ColumnMappingProfile,
    DocumentProfile,
    OutputSchemaProfile,
    TablePostprocessorProfile,
    TableProfile,
    TableTransformationProfile,
)
from app.transformation.region_mapping import STATE_TO_REGION
from app.transformation.electricity_analysis import (
    energy_consumption_to_long,
    maximum_demand_to_long,
)
from app.pipeline.outbreak_profiles import IDSP_WEEKLY_OUTBREAK_REPORT


_REGION_MAPPING = ColumnMappingProfile(
    key_column_position=1,
    target_column_position=0,
    values=STATE_TO_REGION,
)

ENERGY_CONSUMPTION_TABLE = TableProfile(
    name="energy_consumption",
    search_terms=("Energy Consumption",),
    table_index=0,
    transformation=TableTransformationProfile(
        header_row_positions=(1,),
        identity_column_positions=(0, 1),
        measure_column_positions=tuple(range(2, 9)),
        remove_devanagari=True,
        mapping=_REGION_MAPPING,
    ),
    output_filename="clean_energy_consumption.csv",
    output_schema=OutputSchemaProfile(column_count=9, column_levels=1),
    postprocessors=(
        TablePostprocessorProfile(
            name="long",
            processor=energy_consumption_to_long,
            output_filename="energy_consumption_long.csv",
            output_schema=OutputSchemaProfile(
                column_count=4,
                column_levels=1,
                expected_row_count=280,
                required_columns=(
                    "Region",
                    "State",
                    "Date",
                    "Energy_Consumption_MU",
                ),
                non_null_columns=(
                    "Region",
                    "State",
                    "Date",
                    "Energy_Consumption_MU",
                ),
                unique_key_columns=("Region", "State", "Date"),
            ),
        ),
    ),
)

MAXIMUM_DEMAND_TABLE = TableProfile(
    name="maximum_demand",
    search_terms=("Maximum Demand Met",),
    table_index=0,
    transformation=TableTransformationProfile(
        header_row_positions=(1, 2),
        identity_column_positions=(0, 1),
        measure_column_positions=tuple(range(2, 16)),
        fill_merged_headers_from_column=2,
        remove_devanagari=True,
        mapping=_REGION_MAPPING,
    ),
    output_filename="clean_maximum_demand.csv",
    output_schema=OutputSchemaProfile(column_count=16, column_levels=2),
    postprocessors=(
        TablePostprocessorProfile(
            name="long",
            processor=maximum_demand_to_long,
            output_filename="maximum_demand_long.csv",
            output_schema=OutputSchemaProfile(
                column_count=5,
                column_levels=1,
                expected_row_count=273,
                required_columns=(
                    "Region",
                    "State",
                    "Date",
                    "Maximum_Demand_MW",
                    "Peak_Shortage_MW",
                ),
                non_null_columns=(
                    "Region",
                    "State",
                    "Date",
                    "Maximum_Demand_MW",
                    "Peak_Shortage_MW",
                ),
                unique_key_columns=("Region", "State", "Date"),
            ),
        ),
    ),
)

GRID_INDIA_WEEKLY_REPORT = DocumentProfile(
    name="grid_india_weekly_report",
    detection_terms=("Energy Consumption", "Maximum Demand Met"),
    tables=(ENERGY_CONSUMPTION_TABLE, MAXIMUM_DEMAND_TABLE),
)

BUILTIN_DOCUMENT_PROFILES = (
    GRID_INDIA_WEEKLY_REPORT,
    IDSP_WEEKLY_OUTBREAK_REPORT,
)

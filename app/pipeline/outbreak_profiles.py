"""Document profile for IDSP weekly disease-outbreak reports."""

from app.pipeline.profiles import (
    DocumentProfile,
    OutputSchemaProfile,
    TablePostprocessorProfile,
    TableProfile,
    TableTransformationProfile,
)
from app.transformation.outbreak_analysis import (
    current_outbreaks_to_analysis,
    late_outbreaks_to_analysis,
    prepare_current_outbreak_table,
    prepare_late_outbreak_table,
)


_CURRENT_COLUMNS = (
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
_LATE_COLUMNS = (
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


def _schema(columns: tuple[str, ...]) -> OutputSchemaProfile:
    return OutputSchemaProfile(
        column_count=len(columns),
        column_levels=1,
        required_columns=columns,
        non_null_columns=columns,
        unique_key_columns=("Unique_ID",),
    )


CURRENT_OUTBREAKS_TABLE = TableProfile(
    name="current_outbreaks",
    search_terms=("Comments/ Action Taken",),
    page_end_search_terms=(
        "DISEASE OUTBREAKS OF PREVIOUS WEEKS REPORTED LATE",
    ),
    include_end_page=False,
    table_index=0,
    transformation=TableTransformationProfile(
        header_row_positions=(0,),
        identity_column_positions=(0, 1, 2, 3, 6, 7, 8, 9),
        measure_column_positions=(4, 5),
        preprocessor=prepare_current_outbreak_table,
    ),
    output_filename="clean_current_outbreaks.csv",
    output_schema=_schema(_CURRENT_COLUMNS),
    postprocessors=(
        TablePostprocessorProfile(
            name="analysis",
            processor=current_outbreaks_to_analysis,
            output_filename="current_outbreaks_analysis.csv",
            output_schema=_schema(_CURRENT_COLUMNS),
        ),
    ),
)


LATE_OUTBREAKS_TABLE = TableProfile(
    name="late_outbreaks",
    search_terms=(
        "DISEASE OUTBREAKS OF PREVIOUS WEEKS REPORTED LATE",
    ),
    page_end_search_terms=("COVID-19 STATUS",),
    include_end_page=True,
    table_index=0,
    transformation=TableTransformationProfile(
        header_row_positions=(0,),
        identity_column_positions=(0, 1, 2, 3, 6, 7, 8),
        measure_column_positions=(4, 5),
        preprocessor=prepare_late_outbreak_table,
    ),
    output_filename="clean_late_outbreaks.csv",
    output_schema=_schema(_LATE_COLUMNS),
    postprocessors=(
        TablePostprocessorProfile(
            name="analysis",
            processor=late_outbreaks_to_analysis,
            output_filename="late_outbreaks_analysis.csv",
            output_schema=_schema(_LATE_COLUMNS),
        ),
    ),
)


IDSP_WEEKLY_OUTBREAK_REPORT = DocumentProfile(
    name="idsp_weekly_outbreak_report",
    detection_terms=(
        "WEEKLY OUTBREAK REPORT",
        "Integrated Disease Surveillance Program",
    ),
    tables=(CURRENT_OUTBREAKS_TABLE, LATE_OUTBREAKS_TABLE),
)

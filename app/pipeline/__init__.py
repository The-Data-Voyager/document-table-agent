"""Profile-driven document table pipelines."""

from app.pipeline.profiles import (
    ColumnMappingProfile,
    DocumentProfile,
    OutputSchemaProfile,
    TablePostprocessorProfile,
    TableProfile,
    TableTransformationProfile,
)
from app.pipeline.runner import (
    AmbiguousDocumentProfileError,
    DocumentPipelineResult,
    TablePipelineResult,
    UnknownDocumentProfileError,
    detect_document_profile,
    run_document_profile,
    run_profiled_pipeline,
    transform_dataframe_with_profile,
)

__all__ = [
    "AmbiguousDocumentProfileError",
    "ColumnMappingProfile",
    "DocumentPipelineResult",
    "DocumentProfile",
    "OutputSchemaProfile",
    "TablePostprocessorProfile",
    "TablePipelineResult",
    "TableProfile",
    "TableTransformationProfile",
    "UnknownDocumentProfileError",
    "detect_document_profile",
    "run_document_profile",
    "run_profiled_pipeline",
    "transform_dataframe_with_profile",
]

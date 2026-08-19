"""Column-driven analysis helpers for profile-free cleaned tables."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence

import pandas as pd

from app.agent.natural_language import (
    DimensionSemantic,
    MetricSemantic,
    NaturalLanguageQueryResult,
    NaturalLanguageTableAgent,
    SemanticCatalog,
)


_UNIT_WORDS = {
    "d",
    "day",
    "days",
    "hz",
    "mu",
    "mw",
    "pct",
    "percent",
    "percentage",
    "y",
    "year",
    "years",
}
_ALIAS_STOP_WORDS = {
    "name",
    "no",
    "number",
    "of",
    "the",
    "value",
    "values",
}
_IDENTIFIER_CUES = (
    "unique id",
    "identifier",
    "serial no",
    "serial number",
    "s no",
    "sl no",
)
_PREFERRED_CATEGORY_CUES = (
    "state",
    "region",
    "district",
    "disease",
    "illness",
    "category",
    "country",
    "city",
    "date",
    "status",
)


def _normalized_label(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _aliases_for_column(column: object) -> tuple[str, ...]:
    full = _normalized_label(column)
    candidates = [full]
    if "|" in str(column):
        candidates.append(_normalized_label(str(column).split("|")[-1]))
    words = full.split()
    semantic_words = [
        word
        for word in words
        if word not in _ALIAS_STOP_WORDS and word not in _UNIT_WORDS
    ]
    if semantic_words:
        candidates.append(" ".join(semantic_words))
        candidates.extend(semantic_words)
    if words and words[-1] in _UNIT_WORDS and len(words) > 1:
        candidates.append(" ".join(words[:-1]))
    if words and words[-1] not in _UNIT_WORDS:
        candidates.append(words[-1])
    aliases = tuple(dict.fromkeys(item for item in candidates if item))
    return aliases or ("value",)


def _category_priority(column: object) -> tuple[int, int]:
    normalized = _normalized_label(column)
    if normalized == "id" or any(
        cue in normalized for cue in _IDENTIFIER_CUES
    ):
        return (100, 0)
    for position, cue in enumerate(_PREFERRED_CATEGORY_CUES):
        if cue in normalized:
            return (position, 0)
    return (50, len(normalized))


def _numeric_series(series: pd.Series) -> pd.Series | None:
    if pd.api.types.is_bool_dtype(series):
        return None
    if pd.api.types.is_numeric_dtype(series):
        return series

    text = series.map(
        lambda value: "" if pd.isna(value) else str(value).strip()
    )
    nonempty = text.ne("")
    if not nonempty.any():
        return None
    normalized = text.str.replace(",", "", regex=False)
    percentages = normalized.str.endswith("%")
    normalized = normalized.str.removesuffix("%")
    converted = pd.to_numeric(normalized, errors="coerce")
    if converted[nonempty].notna().mean() < 0.8:
        return None
    if percentages.any():
        converted.loc[percentages & converted.notna()] /= 100
    return converted


def prepare_table_for_analysis(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with reliably numeric text columns converted."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    if dataframe.empty or dataframe.shape[1] == 0:
        raise ValueError("The cleaned table has no data to analyze.")

    prepared = dataframe.copy(deep=True)
    prepared.columns = [str(column) for column in prepared.columns]
    for column in prepared.columns:
        converted = _numeric_series(prepared[column])
        if converted is not None:
            prepared[column] = converted
    return prepared


def build_generic_semantic_catalog(
    dataframe: pd.DataFrame,
    *,
    table_name: str = "cleaned_table",
) -> tuple[pd.DataFrame, SemanticCatalog]:
    """Infer a conservative semantic catalog from flat column names."""

    prepared = prepare_table_for_analysis(dataframe)
    numeric_columns = [
        column
        for column in prepared.columns
        if pd.api.types.is_numeric_dtype(prepared[column])
        and not pd.api.types.is_bool_dtype(prepared[column])
    ]
    if not numeric_columns:
        raise ValueError(
            "No reliably numeric columns were found. Correct the table or "
            "convert at least one measure column before asking calculations."
        )

    display_columns = tuple(prepared.columns)
    metrics = []
    for column in numeric_columns:
        normalized = _normalized_label(column)
        default_aggregation = (
            "mean"
            if any(
                cue in normalized
                for cue in ("average", "mean", "age", "rate", "percent")
            )
            else "sum"
        )
        metrics.append(
            MetricSemantic(
                name=normalized or column,
                table=table_name,
                column=column,
                aliases=_aliases_for_column(column),
                default_aggregation=default_aggregation,
                display_columns=display_columns,
            )
        )

    dimension_columns = [
        column for column in prepared.columns if column not in numeric_columns
    ]
    if not dimension_columns:
        dimension_columns = [prepared.columns[0]]
    dimensions = tuple(
        DimensionSemantic(
            column=column,
            aliases=_aliases_for_column(column),
        )
        for column in dimension_columns
    )
    catalog = SemanticCatalog(
        name=f"generic:{table_name}",
        metrics=tuple(metrics),
        dimensions=dimensions,
    )
    return prepared, catalog


def generic_question_examples(dataframe: pd.DataFrame) -> tuple[str, ...]:
    """Build safe example questions from the inferred schema."""

    try:
        prepared, catalog = build_generic_semantic_catalog(dataframe)
    except (TypeError, ValueError):
        return ()
    metric = catalog.metrics[0]
    metric_label = str(metric.column)
    examples = [f"What is the average {metric_label}?"]
    text_dimensions = sorted(
        (
            item
            for item in catalog.dimensions
            if item.column != metric.column
        ),
        key=lambda item: _category_priority(item.column),
    )
    if text_dimensions:
        dimension = text_dimensions[0].column
        examples.append(f"Show top 5 {dimension} by {metric_label}")
        examples.append(f"Calculate total {metric_label} by {dimension}")
    return tuple(examples)


def preferred_category_column(
    dataframe: pd.DataFrame,
    *,
    excluded_columns: Sequence[str] = (),
) -> str | None:
    """Choose a useful default chart category while avoiding identifier IDs."""

    prepared = prepare_table_for_analysis(dataframe)
    excluded = set(excluded_columns)
    candidates = [
        column
        for column in prepared.columns
        if column not in excluded
        and not pd.api.types.is_numeric_dtype(prepared[column])
    ]
    if not candidates:
        candidates = [
            column for column in prepared.columns if column not in excluded
        ]
    if not candidates:
        return None
    return min(candidates, key=_category_priority)


def ask_generic_table_question(
    dataframe: pd.DataFrame,
    question: str,
) -> NaturalLanguageQueryResult:
    """Answer a supported numerical question using inferred column semantics."""

    prepared, catalog = build_generic_semantic_catalog(dataframe)
    return NaturalLanguageTableAgent(
        {"cleaned_table": prepared},
        catalog,
    ).ask(question)


def apply_column_corrections(
    dataframe: pd.DataFrame,
    *,
    dropped_columns: Sequence[str] = (),
    renamed_columns: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Apply validated column removal and renaming without mutating input."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    result = dataframe.copy(deep=True)
    dropped = tuple(dropped_columns)
    unknown = set(dropped) - set(result.columns)
    if unknown:
        raise ValueError(f"Unknown columns selected for removal: {sorted(unknown)!r}.")
    result = result.drop(columns=list(dropped))

    requested = dict(renamed_columns or {})
    unknown = set(requested) - set(result.columns)
    if unknown:
        raise ValueError(f"Unknown columns selected for renaming: {sorted(unknown)!r}.")
    normalized = {
        source: str(target).strip()
        for source, target in requested.items()
        if str(target).strip() and str(target).strip() != str(source)
    }
    final_columns = [normalized.get(column, str(column)) for column in result.columns]
    if len(final_columns) != len(set(final_columns)):
        raise ValueError("Corrected column names must be unique.")
    result.columns = final_columns
    return result.reset_index(drop=True)


def build_chart_data(
    dataframe: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    aggregation: str = "sum",
    limit: int = 50,
) -> pd.DataFrame:
    """Prepare a bounded two-column dataset for Streamlit charts."""

    prepared = prepare_table_for_analysis(dataframe)
    if x_column not in prepared.columns or y_column not in prepared.columns:
        raise ValueError("Chart columns must exist in the cleaned table.")
    if not pd.api.types.is_numeric_dtype(prepared[y_column]):
        raise ValueError(f"{y_column!r} is not reliably numeric.")
    if aggregation not in {"sum", "mean", "min", "max", "none"}:
        raise ValueError("Unsupported chart aggregation.")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer.")

    chart = prepared.loc[:, [x_column, y_column]].dropna().copy()
    if aggregation != "none":
        chart = (
            chart.groupby(x_column, dropna=False, sort=False)[y_column]
            .agg(aggregation)
            .reset_index()
        )
    return chart.head(limit).reset_index(drop=True)

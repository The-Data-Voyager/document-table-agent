"""Local Streamlit interface for the document table agent."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Streamlit Cloud runs a nested entrypoint with its directory at the front of
# sys.path. Add the repository root so absolute ``app.*`` imports work both
# locally and in that deployment environment.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.generic_analysis import (
    apply_column_corrections,
    ask_generic_table_question,
    build_chart_data,
    generic_question_examples,
    preferred_category_column,
    prepare_table_for_analysis,
)
from app.document_service import (
    MAX_UPLOAD_BYTES,
    analysis_tables,
    ask_document_question,
    batch_downloadable_tables,
    build_download_zip,
    dataframe_csv_bytes,
    discover_pdf_tables_automatically,
    discover_pdf_tables_by_keyword,
    discover_pdf_tables_by_page,
    downloadable_tables,
    extract_pdf_table_span,
    inspect_pdf_bytes,
    process_pdf_batch,
    process_pdf_bytes,
)
from app.extraction.generic_extractor import (
    clean_generic_table,
    find_suspicious_columns,
    split_table_sections,
    suggest_header_rows,
)


PROFILE_LABELS = {
    "grid_india_weekly_report": "GRID-INDIA weekly electricity report",
    "idsp_weekly_outbreak_report": "IDSP weekly outbreak report",
}

QUESTION_EXAMPLES = {
    "grid_india_weekly_report": (
        "Which state had the highest energy consumption?",
        "Show the top 5 states by maximum demand",
        "Calculate average peak shortage by region",
    ),
    "idsp_weekly_outbreak_report": (
        "Which state had the most outbreak cases?",
        "Which disease had the most deaths?",
        "Show late reported cases by state",
    ),
}


st.set_page_config(
    page_title="Document Table Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
      .stApp { background: #f7f8f5; }
      [data-testid="stHeader"] { background: rgba(247, 248, 245, 0.92); }
      .hero {
        padding: 1.6rem 1.8rem;
        border: 1px solid #dce4dc;
        border-radius: 18px;
        background: linear-gradient(135deg, #ffffff 0%, #eef5ef 100%);
        margin-bottom: 1.2rem;
      }
      .hero-kicker {
        color: #32734b;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
      }
      .hero h1 { margin: 0.35rem 0 0.4rem; color: #17251c; }
      .hero p { color: #526057; margin: 0; max-width: 760px; }
      div[data-testid="stMetric"] {
        border: 1px solid #dce4dc;
        border-radius: 14px;
        padding: 0.8rem 1rem;
        background: #ffffff;
      }
      div[data-testid="stFileUploader"] {
        border: 1px solid #dce4dc;
        border-radius: 14px;
        padding: 0.7rem;
        background: #ffffff;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, max_entries=6)
def _process_cached(pdf_bytes: bytes):
    return process_pdf_bytes(pdf_bytes)


@st.cache_data(show_spinner=False, max_entries=12)
def _inspect_cached(pdf_bytes: bytes):
    return inspect_pdf_bytes(pdf_bytes)


@st.cache_data(show_spinner=False, max_entries=12)
def _discover_pages_cached(
    pdf_bytes: bytes,
    start_page: int,
    end_page: int,
):
    return discover_pdf_tables_by_page(
        pdf_bytes,
        start_page=start_page,
        end_page=end_page,
    )


@st.cache_data(show_spinner=False, max_entries=12)
def _discover_all_cached(pdf_bytes: bytes):
    return discover_pdf_tables_automatically(pdf_bytes)


@st.cache_data(show_spinner=False, max_entries=12)
def _discover_keyword_cached(pdf_bytes: bytes, keyword: str):
    return discover_pdf_tables_by_keyword(pdf_bytes, keyword)


@st.cache_data(show_spinner=False, max_entries=4)
def _process_batch_cached(documents: tuple[tuple[str, bytes], ...]):
    return process_pdf_batch(documents)


@st.cache_data(show_spinner=False, max_entries=12)
def _extract_span_cached(
    pdf_bytes: bytes,
    start_page: int,
    end_page: int,
    table_index: int,
):
    return extract_pdf_table_span(
        pdf_bytes,
        start_page=start_page,
        end_page=end_page,
        table_index=table_index,
    )


def _profile_label(profile_name: str) -> str:
    return PROFILE_LABELS.get(profile_name, profile_name.replace("_", " ").title())


def _page_label(page_numbers: tuple[int, ...]) -> str:
    if len(page_numbers) == 1:
        return str(page_numbers[0])
    return f"{page_numbers[0]}–{page_numbers[-1]}"


def _validation_details(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_rows": report["rows"],
        "raw_columns": report["columns"],
        "missing_cells": report["missing_cells"],
        "duplicate_rows": report["duplicate_rows"],
        "empty_rows": report["empty_rows"],
        "empty_columns": [str(item) for item in report["empty_columns"]],
        "possible_title_rows": report["possible_title_rows"],
        "possible_header_rows": report["possible_header_rows"],
        "date_like_cells": report["date_like_cells"],
        "possible_hierarchical_header": report[
            "possible_hierarchical_header"
        ],
        "possible_merged_cell_columns": [
            str(item) for item in report["possible_merged_cell_columns"]
        ],
        "missing_values_by_column": {
            str(column): count
            for column, count in report["missing_values_by_column"].items()
        },
    }


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("How it works")
        st.markdown(
            "1. Upload one PDF or a batch.\n"
            "2. Use a profile, scan every page, or locate by page/title.\n"
            "3. Clean repeated headers and correct rare extraction errors.\n"
            "4. Ask questions, build charts, and download corrected tables."
        )
        st.divider()
        st.subheader("High-accuracy profiles")
        st.markdown(
            "- GRID-INDIA weekly electricity reports\n"
            "- IDSP weekly outbreak reports"
        )
        st.caption(
            "Other text-based PDFs can use whole-document or guided "
            "extraction with pdfplumber. Image-only PDFs still require OCR."
        )


def _table_fingerprint(dataframe: pd.DataFrame) -> str:
    payload = dataframe.to_csv(index=False).encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()[:12]


def _render_manual_corrections(
    dataframe: pd.DataFrame,
    *,
    workspace_key: str,
) -> pd.DataFrame:
    """Render non-destructive column, row, and cell correction controls."""

    suspicious = find_suspicious_columns(dataframe)
    if suspicious:
        st.warning(
            "Possible clipped or split source columns were detected: "
            f"{', '.join(suspicious)}. Compare them with the PDF before "
            "keeping, renaming, or removing them."
        )
    st.caption(
        "Remove unwanted columns, edit column names, then correct individual "
        "cells or delete/add rows. The original extraction remains unchanged."
    )
    dropped = st.multiselect(
        "Columns to remove",
        options=[str(column) for column in dataframe.columns],
        key=f"drop-columns-{workspace_key}",
    )
    if len(dropped) == dataframe.shape[1]:
        st.error("Keep at least one column in the corrected table.")
        dropped = []

    remaining = [
        str(column) for column in dataframe.columns if str(column) not in dropped
    ]
    rename_frame = pd.DataFrame(
        {"Original": remaining, "Corrected name": remaining}
    )
    edited_names = st.data_editor(
        rename_frame,
        hide_index=True,
        disabled=["Original"],
        width="stretch",
        key=f"rename-columns-{workspace_key}",
    )
    renamed = dict(
        zip(
            edited_names["Original"].tolist(),
            edited_names["Corrected name"].tolist(),
        )
    )
    try:
        corrected_columns = apply_column_corrections(
            dataframe,
            dropped_columns=dropped,
            renamed_columns=renamed,
        )
    except (TypeError, ValueError) as error:
        st.error(str(error))
        corrected_columns = dataframe.copy(deep=True)

    corrected = st.data_editor(
        corrected_columns,
        hide_index=True,
        num_rows="dynamic",
        width="stretch",
        height=430,
        key=f"edit-cells-{workspace_key}",
    )
    return corrected.reset_index(drop=True)


def _render_generic_question_workspace(
    dataframe: pd.DataFrame,
    *,
    workspace_key: str,
) -> None:
    examples = generic_question_examples(dataframe)
    st.write(
        "Ask numerical questions using the cleaned column names. Results are "
        "calculated locally from the corrected table without an API key."
    )
    if examples:
        st.caption("Try: " + "  •  ".join(examples))
    else:
        st.info(
            "No reliably numeric measure was detected. Correct the table so "
            "at least one measure column contains numbers."
        )
        return

    fingerprint = _table_fingerprint(dataframe)
    answer_key = f"generic-answer-{workspace_key}-{fingerprint}"
    error_key = f"generic-answer-error-{workspace_key}-{fingerprint}"
    with st.form(key=f"generic-question-form-{workspace_key}-{fingerprint}"):
        question = st.text_input(
            "Question about this table",
            placeholder=examples[0],
        )
        submitted = st.form_submit_button("Analyze table", type="primary")
    if submitted:
        try:
            st.session_state[answer_key] = ask_generic_table_question(
                dataframe,
                question,
            )
            st.session_state.pop(error_key, None)
        except (TypeError, ValueError) as error:
            st.session_state.pop(answer_key, None)
            st.session_state[error_key] = str(error)

    if error_key in st.session_state:
        st.error(st.session_state[error_key])
    if answer_key not in st.session_state:
        return
    result = st.session_state[answer_key]
    st.success(
        f"Matched {result.query_result.matched_row_count:,} rows and returned "
        f"{result.query_result.returned_row_count:,} answer rows."
    )
    st.dataframe(result.answer, hide_index=True, width="stretch")
    with st.expander("How this question was interpreted"):
        for note in result.interpretation.notes:
            st.caption(note)
        st.json(asdict(result.interpretation.request))
    with st.expander("Evidence rows"):
        st.dataframe(
            result.evidence,
            hide_index=True,
            width="stretch",
            height=320,
        )


def _render_chart_workspace(
    dataframe: pd.DataFrame,
    *,
    workspace_key: str,
) -> None:
    try:
        prepared = prepare_table_for_analysis(dataframe)
    except (TypeError, ValueError) as error:
        st.info(str(error))
        return
    numeric_columns = [
        column
        for column in prepared.columns
        if pd.api.types.is_numeric_dtype(prepared[column])
        and not pd.api.types.is_bool_dtype(prepared[column])
    ]
    if not numeric_columns:
        st.info("No reliably numeric column is available for a chart.")
        return

    chart_columns = st.columns(3)
    y_column = chart_columns[0].selectbox(
        "Measure",
        options=numeric_columns,
        key=f"chart-y-{workspace_key}",
    )
    x_options = [column for column in prepared.columns if column != y_column]
    if not x_options:
        st.info("Add a category or second numeric column to build a chart.")
        return
    preferred_x = preferred_category_column(
        prepared,
        excluded_columns=(y_column,),
    )
    if preferred_x is None:
        preferred_x = x_options[0]
    x_column = chart_columns[1].selectbox(
        "Category / X axis",
        options=x_options,
        index=x_options.index(preferred_x),
        key=f"chart-x-{workspace_key}",
    )
    chart_type = chart_columns[2].selectbox(
        "Chart type",
        options=("Bar", "Line", "Scatter"),
        key=f"chart-type-{workspace_key}",
    )
    aggregation = st.selectbox(
        "Aggregation",
        options=("sum", "mean", "min", "max", "none"),
        format_func=lambda value: value.title(),
        key=f"chart-aggregation-{workspace_key}",
    )
    try:
        chart_data = build_chart_data(
            dataframe,
            x_column=x_column,
            y_column=y_column,
            aggregation=aggregation,
        )
    except (TypeError, ValueError) as error:
        st.error(str(error))
        return
    if chart_data.empty:
        st.info("The selected columns have no chartable values.")
        return
    chart_display = chart_data.rename(
        columns={x_column: "Category", y_column: "Value"}
    )
    if chart_type == "Line":
        st.line_chart(
            chart_display,
            x="Category",
            y="Value",
            x_label=x_column,
            y_label=y_column,
            height=420,
        )
    elif chart_type == "Scatter":
        if not pd.api.types.is_numeric_dtype(prepared[x_column]):
            st.info("Scatter charts require a numeric X axis.")
            return
        st.scatter_chart(
            chart_display,
            x="Category",
            y="Value",
            x_label=x_column,
            y_label=y_column,
            height=420,
        )
    else:
        st.bar_chart(
            chart_display,
            x="Category",
            y="Value",
            x_label=x_column,
            y_label=y_column,
            height=420,
        )


def _render_explore_workspace(
    tables: dict[str, pd.DataFrame],
    *,
    workspace_key: str,
) -> None:
    selected = st.selectbox(
        "Table to explore",
        options=list(tables),
        format_func=lambda name: name.replace("_", " ").title(),
        key=f"explore-table-{workspace_key}",
    )
    source = tables[selected]
    correction_tab, question_tab, chart_tab, download_tab = st.tabs(
        ("Correct", "Ask any table", "Charts", "Download corrected")
    )
    with correction_tab:
        corrected = _render_manual_corrections(
            source,
            workspace_key=f"{workspace_key}-{selected}",
        )
    with question_tab:
        _render_generic_question_workspace(
            corrected,
            workspace_key=f"{workspace_key}-{selected}",
        )
    with chart_tab:
        _render_chart_workspace(
            corrected,
            workspace_key=f"{workspace_key}-{selected}",
        )
    with download_tab:
        st.download_button(
            "Download corrected CSV",
            data=dataframe_csv_bytes(corrected),
            file_name=f"{selected}_corrected.csv",
            mime="text/csv",
            type="primary",
        )


def _render_batch_workspace(
    documents: tuple[tuple[str, bytes], ...],
    *,
    batch_key: str,
) -> None:
    with st.expander("Batch processing", expanded=True):
        st.write(
            "Process every uploaded PDF with a high-accuracy profile when "
            "available, otherwise use automatic whole-document discovery."
        )
        submitted = st.button(
            "Process all PDFs",
            type="primary",
            key=f"process-batch-{batch_key}",
        )
        result_key = f"batch-result-{batch_key}"
        error_key = f"batch-error-{batch_key}"
        if submitted:
            try:
                with st.spinner("Extracting tables from all uploaded PDFs..."):
                    st.session_state[result_key] = _process_batch_cached(documents)
                st.session_state.pop(error_key, None)
            except (TypeError, ValueError) as error:
                st.session_state.pop(result_key, None)
                st.session_state[error_key] = str(error)
        if error_key in st.session_state:
            st.error(st.session_state[error_key])
        if result_key not in st.session_state:
            return

        result = st.session_state[result_key]
        metrics = st.columns(3)
        metrics[0].metric("Documents", len(result.documents))
        metrics[1].metric("Successful", result.successful_documents)
        metrics[2].metric("Tables", result.table_count)
        for document in result.documents:
            if document.error:
                st.error(f"{document.filename}: {document.error}")
                continue
            st.success(
                f"{document.filename}: {len(document.tables)} table(s) via "
                f"{document.method}."
            )
            for warning in document.warnings:
                st.warning(warning)
        if result.table_count:
            downloads = batch_downloadable_tables(result)
            st.download_button(
                "Download all batch tables (.zip)",
                data=build_download_zip(downloads),
                file_name="document_table_batch.zip",
                mime="application/zip",
                type="primary",
            )


def _render_tables_tab(result) -> None:
    tables = analysis_tables(result)
    selected = st.selectbox(
        "Analysis-ready table",
        options=list(tables),
        format_func=lambda name: name.replace("_", " ").title(),
    )
    table = tables[selected]
    left, right = st.columns(2)
    left.metric("Rows", f"{len(table):,}")
    right.metric("Columns", f"{table.shape[1]:,}")
    st.dataframe(table, hide_index=True, width="stretch", height=470)
    with st.expander("Column types"):
        st.dataframe(
            {
                "Column": [str(column) for column in table.columns],
                "Type": [str(dtype) for dtype in table.dtypes],
            },
            hide_index=True,
            width="stretch",
        )


def _render_ask_tab(result, document_key: str) -> None:
    examples = QUESTION_EXAMPLES.get(result.profile_name, ())
    st.write(
        "Ask a supported question in plain English. The interpretation is "
        "deterministic and uses only the extracted table—no API key is needed."
    )
    if examples:
        st.caption("Try: " + "  •  ".join(examples))

    with st.form(key=f"question-form-{document_key}"):
        question = st.text_input(
            "Question",
            placeholder=examples[0] if examples else "Ask about the table",
        )
        submitted = st.form_submit_button("Ask the document", type="primary")

    answer_key = f"answer-{document_key}"
    error_key = f"answer-error-{document_key}"
    if submitted:
        if not question.strip():
            st.session_state.pop(answer_key, None)
            st.session_state[error_key] = "Enter a question first."
        else:
            try:
                st.session_state[answer_key] = ask_document_question(
                    result, question
                )
                st.session_state.pop(error_key, None)
            except (TypeError, ValueError) as error:
                st.session_state.pop(answer_key, None)
                st.session_state[error_key] = str(error)

    if error_key in st.session_state:
        st.error(st.session_state[error_key])
    if answer_key not in st.session_state:
        return

    natural_result = st.session_state[answer_key]
    query_result = natural_result.query_result
    st.success(
        f"Matched {query_result.matched_row_count:,} source rows and returned "
        f"{query_result.returned_row_count:,} answer rows."
    )
    st.dataframe(
        natural_result.answer,
        hide_index=True,
        width="stretch",
    )
    with st.expander("How the question was interpreted"):
        st.write(
            f"Matched metric: **{natural_result.interpretation.metric_name}**"
        )
        for note in natural_result.interpretation.notes:
            st.caption(note)
        st.json(asdict(natural_result.interpretation.request))
    with st.expander("Evidence rows"):
        st.caption(
            "These are the extracted rows used to calculate the answer."
        )
        st.dataframe(
            natural_result.evidence,
            hide_index=True,
            width="stretch",
            height=360,
        )


def _render_validation_tab(result) -> None:
    st.success("All configured transformed-table schema checks passed.")
    st.caption(
        "The observations below describe the raw PDF extraction. Missing cells "
        "and merged headers can be expected before profile transformation."
    )
    for table_name, table_result in result.tables.items():
        with st.expander(
            table_name.replace("_", " ").title(), expanded=True
        ):
            page_text = _page_label(table_result.page_numbers)
            columns = st.columns(4)
            columns[0].metric("Pages", page_text)
            columns[1].metric(
                "Raw shape",
                f"{table_result.raw_table.shape[0]} × "
                f"{table_result.raw_table.shape[1]}",
            )
            columns[2].metric(
                "Clean rows", f"{len(table_result.transformed_table):,}"
            )
            columns[3].metric(
                "Analysis rows",
                f"{sum(len(item) for item in table_result.postprocessed_tables.values()):,}",
            )
            for warning in table_result.warnings:
                st.warning(warning)
            st.json(_validation_details(table_result.validation_report))


def _render_downloads_tab(result) -> None:
    downloads = downloadable_tables(result)
    st.write(
        "Download the cleaned wide tables and analysis-ready tables. CSV files "
        "use UTF-8 encoding that opens cleanly in Excel."
    )
    st.download_button(
        "Download all tables (.zip)",
        data=build_download_zip(downloads),
        file_name=f"{result.profile_name}_tables.zip",
        mime="application/zip",
        type="primary",
    )
    st.divider()
    for index, (filename, table) in enumerate(downloads.items()):
        label_column, button_column = st.columns([3, 1])
        label_column.write(f"**{filename}**")
        label_column.caption(
            f"{len(table):,} rows × {table.shape[1]:,} columns"
        )
        button_column.download_button(
            "Download CSV",
            data=dataframe_csv_bytes(table),
            file_name=filename,
            mime="text/csv",
            key=f"download-{index}-{filename}",
            width="stretch",
        )


def _render_profile_workspace(result, document_key: str) -> None:
    """Render the high-accuracy workflow for a known document profile."""

    st.success(f"Detected profile: {_profile_label(result.profile_name)}")
    analysis = analysis_tables(result)
    summary_columns = st.columns(3)
    summary_columns[0].metric("Tables found", len(result.tables))
    summary_columns[1].metric(
        "Analysis rows", f"{sum(len(table) for table in analysis.values()):,}"
    )
    summary_columns[2].metric(
        "Pages used",
        len(
            {
                page
                for table in result.tables.values()
                for page in table.page_numbers
            }
        ),
    )

    tables_tab, ask_tab, explore_tab, validation_tab, downloads_tab = st.tabs(
        ("Tables", "Curated questions", "Explore & charts", "Validation", "Downloads")
    )
    with tables_tab:
        _render_tables_tab(result)
    with ask_tab:
        _render_ask_tab(result, document_key)
    with explore_tab:
        _render_explore_workspace(
            analysis,
            workspace_key=f"profile-{document_key}",
        )
    with validation_tab:
        _render_validation_tab(result)
    with downloads_tab:
        _render_downloads_tab(result)


def _render_guided_extractor(
    pdf_bytes: bytes,
    document_key: str,
    mode: str,
) -> None:
    """Render profile-free discovery, preview, cleanup, and downloads."""

    try:
        inspection = _inspect_cached(pdf_bytes)
    except Exception as error:
        st.error(f"This PDF could not be inspected: {error}")
        return

    summary = st.columns(2)
    summary[0].metric("PDF pages", inspection.page_count)
    summary[1].metric(
        "Searchable text",
        "Yes" if inspection.has_extractable_text else "No",
    )

    if mode == "Automatic all tables":
        mode_key = "automatic"
    elif mode == "Page or page range":
        mode_key = "pages"
    else:
        mode_key = "keyword"
    discovery_key = f"guided-discovery-{document_key}-{mode_key}"
    error_key = f"guided-error-{document_key}-{mode_key}"

    if mode_key == "automatic":
        st.subheader("Discover every table in the PDF")
        st.caption(
            "This scans all pages with pdfplumber. High-accuracy profile "
            "transformations are not applied, so you can review every raw "
            "candidate before cleaning it."
        )
        if not inspection.has_extractable_text:
            st.warning(
                "This PDF has little or no searchable text. pdfplumber may "
                "find ruled grids, but cell text will require the OCR "
                "fallback planned for scanned pages."
            )
        submitted = st.button(
            "Scan complete PDF",
            type="primary",
            key=f"automatic-search-{document_key}",
        )
        if submitted:
            try:
                with st.spinner("Scanning every page for table grids..."):
                    st.session_state[discovery_key] = _discover_all_cached(
                        pdf_bytes
                    )
                st.session_state.pop(error_key, None)
            except Exception as error:
                st.session_state.pop(discovery_key, None)
                st.session_state[error_key] = str(error)
    elif mode_key == "pages":
        st.subheader("Locate tables by page")
        with st.form(key=f"page-search-{document_key}"):
            page_columns = st.columns(2)
            start_page = int(
                page_columns[0].number_input(
                    "Start page",
                    min_value=1,
                    max_value=inspection.page_count,
                    value=1,
                    step=1,
                )
            )
            end_page = int(
                page_columns[1].number_input(
                    "End page",
                    min_value=1,
                    max_value=inspection.page_count,
                    value=1,
                    step=1,
                )
            )
            submitted = st.form_submit_button("Find tables", type="primary")
        if submitted:
            try:
                with st.spinner("Scanning the selected pages for tables..."):
                    st.session_state[discovery_key] = _discover_pages_cached(
                        pdf_bytes,
                        start_page,
                        end_page,
                    )
                st.session_state.pop(error_key, None)
            except Exception as error:
                st.session_state.pop(discovery_key, None)
                st.session_state[error_key] = str(error)
    else:
        st.subheader("Locate a table by name or keyword")
        if not inspection.has_extractable_text:
            st.warning(
                "This PDF has little or no searchable text. Keyword search "
                "may require OCR; page-number extraction can still be tried."
            )
        with st.form(key=f"keyword-search-{document_key}"):
            keyword = st.text_input(
                "Table name or nearby text",
                placeholder="For example: Energy Consumption",
            )
            submitted = st.form_submit_button(
                "Search and find tables",
                type="primary",
            )
        if submitted:
            try:
                with st.spinner("Searching the PDF and detecting tables..."):
                    st.session_state[discovery_key] = _discover_keyword_cached(
                        pdf_bytes,
                        keyword,
                    )
                st.session_state.pop(error_key, None)
            except Exception as error:
                st.session_state.pop(discovery_key, None)
                st.session_state[error_key] = str(error)

    if error_key in st.session_state:
        st.error(st.session_state[error_key])
    if discovery_key not in st.session_state:
        if mode_key == "automatic":
            st.info("Scan the complete PDF to list every detected table grid.")
        else:
            st.info(
                "Enter the locator details above, then find the available "
                "tables."
            )
        return

    discovery = st.session_state[discovery_key]
    if not discovery.searched_pages:
        st.warning(
            f"No pages contained {discovery.keyword!r}. Try fewer words, a "
            "nearby heading, or locate the table by page number."
        )
        return
    if not discovery.candidates:
        pages = ", ".join(str(page) for page in discovery.searched_pages)
        st.warning(
            f"No grid tables were detected on page(s) {pages}. The table may "
            "be an image or may require different extraction settings."
        )
        return

    logical_count = discovery.logical_table_count
    if logical_count == len(discovery.candidates):
        count_description = f"{logical_count} logical table(s)"
    else:
        count_description = (
            f"{logical_count} logical table(s) inside "
            f"{len(discovery.candidates)} detected grid(s)"
        )
    st.success(
        f"Found {count_description} across "
        f"{len(discovery.searched_pages)} scanned page(s)."
    )
    if mode_key == "automatic" and discovery.pages_without_candidates:
        missing_pages = ", ".join(
            str(page) for page in discovery.pages_without_candidates
        )
        st.caption(
            f"No table grid was detected on page(s): {missing_pages}. These "
            "pages may contain no tables, borderless tables, or scanned "
            "content that needs OCR."
        )
    candidate_index = st.selectbox(
        "Select the table to extract",
        options=range(len(discovery.candidates)),
        format_func=lambda index: discovery.candidates[index].label,
        key=f"candidate-{document_key}-{mode_key}",
    )
    candidate = discovery.candidates[candidate_index]
    raw_table = candidate.dataframe
    last_page = candidate.page_number

    if candidate.page_number < discovery.page_count:
        join_pages = st.checkbox(
            "This table continues onto following pages",
            key=f"join-pages-{document_key}-{mode_key}-{candidate_index}",
        )
        if join_pages:
            last_page = int(
                st.number_input(
                    "Continue through page",
                    min_value=candidate.page_number + 1,
                    max_value=discovery.page_count,
                    value=candidate.page_number + 1,
                    step=1,
                    key=(
                        f"span-end-{document_key}-{mode_key}-"
                        f"{candidate_index}"
                    ),
                )
            )
            try:
                raw_table = _extract_span_cached(
                    pdf_bytes,
                    candidate.page_number,
                    last_page,
                    candidate.table_index,
                )
            except Exception as error:
                st.error(
                    "The same table number could not be joined across that "
                    f"page range: {error}"
                )
                return

    sections = split_table_sections(raw_table)
    section_number = 1
    if len(sections) > 1:
        st.warning(
            f"This detected grid contains {len(sections)} vertically stacked "
            "tables. Select the one you want before preparing its headers."
        )
        section_index = st.selectbox(
            "Table section on this page",
            options=range(len(sections)),
            format_func=lambda index: sections[index].label,
            key=f"section-{document_key}-{mode_key}-{candidate_index}",
        )
        section_number = section_index + 1
        raw_table = sections[section_index].dataframe

    st.subheader("Prepare the extracted table")
    suggested_headers = suggest_header_rows(raw_table)
    if suggested_headers:
        st.info(
            f"Suggested header: start at row {suggested_headers[0] + 1} and "
            f"combine {len(suggested_headers)} row(s)."
        )
    controls = st.columns(2)
    header_options = [None, *range(min(15, len(raw_table)))]
    suggested_start = suggested_headers[0] if suggested_headers else None
    header_row = controls[0].selectbox(
        "Header starts at",
        options=header_options,
        format_func=lambda row: (
            "No header - keep all rows" if row is None else f"Row {row + 1}"
        ),
        index=header_options.index(suggested_start),
        key=f"header-{document_key}-{mode_key}-{candidate_index}",
    )
    if header_row is None:
        header_row_count = 1
        controls[1].caption(
            "Select a starting row to combine multi-row or merged headers."
        )
    else:
        maximum_header_rows = min(4, len(raw_table) - header_row)
        suggested_count = (
            len(suggested_headers)
            if suggested_headers and header_row == suggested_headers[0]
            else 1
        )
        header_row_count = int(
            controls[1].number_input(
                "Header rows to combine",
                min_value=1,
                max_value=maximum_header_rows,
                value=min(suggested_count, maximum_header_rows),
                step=1,
                key=(
                    f"header-count-{document_key}-{mode_key}-"
                    f"{candidate_index}"
                ),
            )
        )

    cleanup_controls = st.columns(2)
    remove_devanagari = cleanup_controls[0].checkbox(
        "Remove Hindi/Devanagari text from clean output",
        value=True,
        key=f"remove-devanagari-{document_key}-{mode_key}-{candidate_index}",
    )
    drop_empty = cleanup_controls[1].checkbox(
        "Remove completely empty rows and columns",
        value=True,
        key=f"drop-empty-{document_key}-{mode_key}-{candidate_index}",
    )
    continuation_controls = st.columns(2)
    remove_repeated_headers = continuation_controls[0].checkbox(
        "Remove repeated headers on continuation pages",
        value=True,
        key=f"repeated-headers-{document_key}-{mode_key}-{candidate_index}",
    )
    merge_continuation_rows = continuation_controls[1].checkbox(
        "Merge wrapped text rows into the preceding record",
        value=True,
        key=f"merge-continuations-{document_key}-{mode_key}-{candidate_index}",
        help=(
            "Repairs long Comments, Remarks, Notes, or Description cells that "
            "pdfplumber emitted as sparse extra rows."
        ),
    )
    cleaned_table = clean_generic_table(
        raw_table,
        header_row=header_row,
        header_row_count=header_row_count,
        drop_empty=drop_empty,
        remove_devanagari=remove_devanagari,
        remove_repeated_headers=remove_repeated_headers,
        merge_continuation_rows=merge_continuation_rows,
    )

    raw_tab, clean_tab, correction_tab, question_tab, chart_tab, download_tab = st.tabs(
        ("Raw layout", "Clean preview", "Correct", "Ask", "Charts", "Downloads")
    )
    with raw_tab:
        st.caption(
            "Raw cell positions are preserved. Blank cells can represent "
            "merged PDF headers; use Clean preview to see them reconstructed."
        )
        st.dataframe(raw_table, hide_index=True, width="stretch", height=440)
    with clean_tab:
        st.caption(
            "Multi-row headers are forward-filled and flattened with a | "
            "separator for CSV compatibility."
        )
        st.dataframe(
            cleaned_table,
            hide_index=True,
            width="stretch",
            height=440,
        )
    correction_key = (
        f"guided-{document_key}-{mode_key}-{candidate_index}-{section_number}"
    )
    with correction_tab:
        corrected_table = _render_manual_corrections(
            cleaned_table,
            workspace_key=correction_key,
        )
    with question_tab:
        _render_generic_question_workspace(
            corrected_table,
            workspace_key=correction_key,
        )
    with chart_tab:
        _render_chart_workspace(
            corrected_table,
            workspace_key=correction_key,
        )
    with download_tab:
        page_suffix = (
            f"page_{candidate.page_number}"
            if last_page == candidate.page_number
            else f"pages_{candidate.page_number}_{last_page}"
        )
        file_base = f"{page_suffix}_table_{candidate.table_index + 1}"
        if len(sections) > 1:
            file_base = f"{file_base}_section_{section_number}"
        download_columns = st.columns(2)
        download_columns[0].download_button(
            "Download raw CSV",
            data=raw_table.to_csv(index=False, header=False).encode("utf-8-sig"),
            file_name=f"{file_base}_raw.csv",
            mime="text/csv",
            width="stretch",
        )
        download_columns[1].download_button(
            "Download corrected CSV",
            data=dataframe_csv_bytes(corrected_table),
            file_name=f"{file_base}_corrected.csv",
            mime="text/csv",
            width="stretch",
        )


def main() -> None:
    _render_sidebar()
    st.markdown(
        """
        <section class="hero">
          <div class="hero-kicker">Extract • correct • analyze</div>
          <h1>Document Table Agent</h1>
          <p>Turn one PDF or a batch into reviewable tables, correct rare
          extraction mistakes, ask data questions, create charts, and export
          clean CSV files.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload one or more PDF reports",
        type=("pdf",),
        accept_multiple_files=True,
        help=(
            f"Maximum size per file: {MAX_UPLOAD_BYTES // (1024 * 1024)} MB; "
            "up to 10 files per batch."
        ),
    )
    if not uploaded_files:
        st.info(
            "Upload a text-based PDF. Known report layouts use high-accuracy "
            "profiles; other PDFs can be scanned automatically or explored "
            "by page number or title."
        )
        return

    documents = tuple(
        (uploaded.name, uploaded.getvalue()) for uploaded in uploaded_files
    )
    batch_key = hashlib.sha256(
        b"".join(
            name.encode("utf-8", errors="replace") + payload
            for name, payload in documents
        )
    ).hexdigest()[:16]
    if len(documents) > 1:
        _render_batch_workspace(documents, batch_key=batch_key)
        selected_index = st.selectbox(
            "Document to inspect interactively",
            options=range(len(documents)),
            format_func=lambda index: documents[index][0],
        )
    else:
        selected_index = 0
    uploaded_name, pdf_bytes = documents[selected_index]
    document_key = hashlib.sha256(pdf_bytes).hexdigest()[:16]
    st.caption(f"Interactive file: {uploaded_name}")
    mode = st.radio(
        "How should the app locate the table?",
        options=(
            "High-accuracy profile",
            "Automatic all tables",
            "Page or page range",
            "Table name or keyword",
        ),
        horizontal=True,
    )
    if mode != "High-accuracy profile":
        _render_guided_extractor(pdf_bytes, document_key, mode)
        return

    try:
        with st.spinner("Detecting the layout and extracting tables..."):
            result = _process_cached(pdf_bytes)
    except Exception as error:
        st.error(f"No high-accuracy profile could process this PDF: {error}")
        st.caption(
            "Choose Automatic all tables, Page or page range, or Table name "
            "or keyword above to extract from an unregistered layout."
        )
        return
    _render_profile_workspace(result, document_key)


if __name__ == "__main__":
    main()

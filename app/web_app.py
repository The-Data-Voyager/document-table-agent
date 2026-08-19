"""Local Streamlit interface for the document table agent."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any

import streamlit as st

from app.document_service import (
    MAX_UPLOAD_BYTES,
    analysis_tables,
    ask_document_question,
    build_download_zip,
    dataframe_csv_bytes,
    downloadable_tables,
    process_pdf_bytes,
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
            "1. Upload a supported PDF.\n"
            "2. The layout profile is detected automatically.\n"
            "3. Inspect, question, and download the cleaned tables."
        )
        st.divider()
        st.subheader("Supported document families")
        st.markdown(
            "- GRID-INDIA weekly electricity reports\n"
            "- IDSP weekly outbreak reports"
        )
        st.caption(
            "A new PDF layout needs a matching profile. The app will not "
            "silently guess an unknown layout."
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


def main() -> None:
    _render_sidebar()
    st.markdown(
        """
        <section class="hero">
          <div class="hero-kicker">Profile-driven PDF analysis</div>
          <h1>Document Table Agent</h1>
          <p>Upload a supported report, inspect the cleaned tables, ask data
          questions, and download verified CSV outputs.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload a PDF report",
        type=("pdf",),
        help=f"Maximum size: {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
    )
    if uploaded is None:
        st.info(
            "Start by uploading either a GRID-INDIA weekly report or an IDSP "
            "weekly outbreak report."
        )
        return

    pdf_bytes = uploaded.getvalue()
    document_key = hashlib.sha256(pdf_bytes).hexdigest()[:16]
    try:
        with st.spinner("Detecting the layout and extracting tables…"):
            result = _process_cached(pdf_bytes)
    # This is the UI boundary: corrupt PDFs and parser-specific failures should
    # become a useful message instead of an uncaught Streamlit exception.
    except Exception as error:
        st.error(f"This PDF could not be processed: {error}")
        st.caption(
            "If this is a new layout, add a document profile before processing it."
        )
        return

    st.caption(f"File: {uploaded.name}")
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

    tables_tab, ask_tab, validation_tab, downloads_tab = st.tabs(
        ("Tables", "Ask", "Validation", "Downloads")
    )
    with tables_tab:
        _render_tables_tab(result)
    with ask_tab:
        _render_ask_tab(result, document_key)
    with validation_tab:
        _render_validation_tab(result)
    with downloads_tab:
        _render_downloads_tab(result)


if __name__ == "__main__":
    main()

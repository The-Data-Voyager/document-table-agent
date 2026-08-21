from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_web_app_starts_in_its_empty_upload_state():
    app = AppTest.from_file("streamlit_app.py").run(timeout=20)

    assert not app.exception
    assert any("Document Table Agent" in item.value for item in app.markdown)
    assert "Upload a text-based PDF" in app.info[0].value
    styles = next(
        item.value for item in app.markdown if ".hero" in item.value
    )
    assert "#f7f8f5" not in styles
    assert "background: #ffffff" not in styles
    assert "color: inherit" in styles


def test_web_components_render_a_processed_document():
    sample_pdf = next(
        path.resolve()
        for path in Path("sample_documents").glob("*.pdf")
        if path.name.startswith("Weekly ")
    )
    script = f"""
from pathlib import Path
from app.document_service import process_pdf_bytes
from app.web_app import (
    _render_ask_tab,
    _render_downloads_tab,
    _render_tables_tab,
    _render_validation_tab,
)

result = process_pdf_bytes(Path({str(sample_pdf)!r}).read_bytes())
_render_tables_tab(result)
_render_ask_tab(result, "test-document")
_render_validation_tab(result)
_render_downloads_tab(result)
"""

    app = AppTest.from_string(script).run(timeout=30)

    assert not app.exception
    assert len(app.dataframe) >= 2
    assert any("schema checks passed" in item.value for item in app.success)


def test_guided_extractor_renders_discovered_table_candidates():
    sample_pdf = Path(
        "sample_documents/75788759701752062509.pdf"
    ).resolve()
    script = f"""
from pathlib import Path
import streamlit as st
from app.document_service import discover_pdf_tables_by_page
from app.web_app import _render_guided_extractor

pdf_bytes = Path({str(sample_pdf)!r}).read_bytes()
st.session_state["guided-discovery-test-document-pages"] = (
    discover_pdf_tables_by_page(pdf_bytes, start_page=3, end_page=3)
)
_render_guided_extractor(
    pdf_bytes,
    "test-document",
    "Page or page range",
)
"""

    app = AppTest.from_string(script).run(timeout=30)

    assert not app.exception
    assert len(app.dataframe) >= 2
    assert any("logical table(s)" in item.value for item in app.success)
    assert any(
        item.label == "Merge wrapped text rows into the preceding record"
        for item in app.checkbox
    )


def test_guided_cleanup_error_is_contained_and_raw_table_stays_downloadable():
    sample_pdf = Path(
        "sample_documents/75788759701752062509.pdf"
    ).resolve()
    script = f"""
from pathlib import Path
import streamlit as st
from app.document_service import discover_pdf_tables_by_page
from app import web_app

pdf_bytes = Path({str(sample_pdf)!r}).read_bytes()
st.session_state["guided-discovery-cleanup-error-pages"] = (
    discover_pdf_tables_by_page(pdf_bytes, start_page=3, end_page=3)
)
def fail_cleanup(*args, **kwargs):
    raise TypeError("simulated mixed header failure")
web_app.clean_generic_table = fail_cleanup
web_app._render_guided_extractor(
    pdf_bytes,
    "cleanup-error",
    "Page or page range",
)
"""

    app = AppTest.from_string(script).run(timeout=30)

    assert not app.exception
    assert any("could not be prepared" in item.value for item in app.error)
    assert any(
        item.label == "Download raw table"
        for item in app.get("download_button")
    )


def test_automatic_discovery_explains_grid_and_logical_table_counts():
    sample_pdf = next(
        path.resolve()
        for path in Path("sample_documents").glob("*.pdf")
        if path.name.startswith("Weekly ")
    )
    script = f"""
from pathlib import Path
import streamlit as st
from app.document_service import discover_pdf_tables_automatically
from app.web_app import _render_guided_extractor

pdf_bytes = Path({str(sample_pdf)!r}).read_bytes()
st.session_state["guided-discovery-automatic-document-automatic"] = (
    discover_pdf_tables_automatically(pdf_bytes)
)
_render_guided_extractor(
    pdf_bytes,
    "automatic-document",
    "Automatic all tables",
)
"""

    app = AppTest.from_string(script).run(timeout=30)

    assert not app.exception
    assert any("7 logical table(s)" in item.value for item in app.success)
    assert any("5 detected grid(s)" in item.value for item in app.success)
    assert any("Scan complete PDF" == item.label for item in app.button)


def test_guided_extractor_offers_stacked_table_sections():
    sample_pdf = next(
        path.resolve()
        for path in Path("sample_documents").glob("*.pdf")
        if path.name.startswith("Weekly ")
    )
    script = f"""
from pathlib import Path
import streamlit as st
from app.document_service import discover_pdf_tables_by_page
from app.web_app import _render_guided_extractor

pdf_bytes = Path({str(sample_pdf)!r}).read_bytes()
st.session_state["guided-discovery-stacked-document-pages"] = (
    discover_pdf_tables_by_page(pdf_bytes, start_page=2, end_page=2)
)
_render_guided_extractor(
    pdf_bytes,
    "stacked-document",
    "Page or page range",
)
"""

    app = AppTest.from_string(script).run(timeout=30)

    assert not app.exception
    assert any("vertically stacked" in item.value for item in app.warning)
    assert any(
        item.label == "Table section on this page"
        for item in app.selectbox
    )


def test_explore_workspace_offers_corrections_questions_and_charts():
    script = """
import pandas as pd
from app.web_app import _render_explore_workspace

table = pd.DataFrame({
    "State": ["Assam", "Bihar"],
    "Cases": [10, 20],
})
_render_explore_workspace({"outbreaks": table}, workspace_key="explore-test")
"""

    app = AppTest.from_string(script).run(timeout=30)

    assert not app.exception
    assert any(item.label == "Columns to remove" for item in app.multiselect)
    assert any(
        item.label == "Question about this table" for item in app.text_input
    )
    assert any(item.label == "Chart type" for item in app.selectbox)


def test_chart_workspace_uses_safe_fields_for_pdf_column_punctuation():
    script = """
import pandas as pd
from app.web_app import _render_chart_workspace

table = pd.DataFrame({
    "Disease/ Illness": ["Typhoid", "Measles", "Typhoid"],
    "No. of Cases": [9, 5, 15],
})
_render_chart_workspace(table, workspace_key="punctuated-chart")
"""

    app = AppTest.from_string(script).run(timeout=30)

    assert not app.exception
    chart = app.get("arrow_vega_lite_chart")[0]
    assert '"field": "Category"' in chart.proto.spec
    assert '"field": "Value"' in chart.proto.spec
    assert '"title": "No. of Cases"' in chart.proto.spec
    assert '"height": 420' in chart.proto.spec

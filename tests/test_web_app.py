from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_web_app_starts_in_its_empty_upload_state():
    app = AppTest.from_file("streamlit_app.py").run(timeout=20)

    assert not app.exception
    assert any("Document Table Agent" in item.value for item in app.markdown)
    assert "Upload a text-based PDF" in app.info[0].value


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
    assert any("table candidate" in item.value for item in app.success)


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

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_web_app_starts_in_its_empty_upload_state():
    app = AppTest.from_file("app/web_app.py").run(timeout=20)

    assert not app.exception
    assert any("Document Table Agent" in item.value for item in app.markdown)
    assert "Start by uploading" in app.info[0].value


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

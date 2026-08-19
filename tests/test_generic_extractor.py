from pathlib import Path

import pandas as pd

from app.document_service import (
    discover_pdf_tables_by_keyword,
    discover_pdf_tables_by_page,
    extract_pdf_table_span,
    inspect_pdf_bytes,
)
from app.extraction.generic_extractor import clean_generic_table


def _outbreak_pdf_bytes() -> bytes:
    return Path("sample_documents/75788759701752062509.pdf").read_bytes()


def test_guided_page_discovery_returns_table_candidates():
    pdf_bytes = _outbreak_pdf_bytes()

    inspection = inspect_pdf_bytes(pdf_bytes)
    discovery = discover_pdf_tables_by_page(
        pdf_bytes,
        start_page=3,
        end_page=3,
    )

    assert inspection.page_count == 16
    assert inspection.has_extractable_text is True
    assert discovery.searched_pages == (3,)
    assert discovery.candidates
    assert discovery.candidates[0].page_number == 3
    assert "Page 3 - Table 1" in discovery.candidates[0].label


def test_guided_keyword_discovery_locates_matching_pages():
    discovery = discover_pdf_tables_by_keyword(
        _outbreak_pdf_bytes(),
        "Comments/ Action Taken",
    )

    assert 3 in discovery.searched_pages
    assert discovery.keyword == "Comments/ Action Taken"
    assert discovery.candidates


def test_guided_span_extraction_joins_the_same_table_number():
    table = extract_pdf_table_span(
        _outbreak_pdf_bytes(),
        start_page=3,
        end_page=4,
        table_index=0,
    )

    assert table.shape[0] > 20
    assert table.shape[1] >= 10


def test_generic_cleanup_promotes_headers_and_drops_empty_cells():
    raw = pd.DataFrame(
        [
            [" State ", "Cases", None],
            ["Delhi", "10", None],
            [" ", "", None],
        ]
    )

    cleaned = clean_generic_table(raw, header_row=0, drop_empty=True)

    assert cleaned.to_dict(orient="records") == [
        {"State": "Delhi", "Cases": "10"}
    ]


def test_generic_cleanup_makes_duplicate_headers_unique():
    raw = pd.DataFrame([["Value", "Value"], [1, 2]])

    cleaned = clean_generic_table(raw, header_row=0)

    assert cleaned.columns.tolist() == ["Value", "Value_2"]

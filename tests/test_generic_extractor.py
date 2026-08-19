from pathlib import Path

import pandas as pd

from app.document_service import (
    discover_pdf_tables_automatically,
    discover_pdf_tables_by_keyword,
    discover_pdf_tables_by_page,
    extract_pdf_table_span,
    inspect_pdf_bytes,
)
from app.extraction.generic_extractor import clean_generic_table
from app.extraction.generic_extractor import (
    discover_table_candidates,
    split_table_sections,
    suggest_header_rows,
)


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


def test_automatic_discovery_scans_all_pages_and_counts_stacked_tables():
    electricity_pdf = next(
        path
        for path in Path("sample_documents").glob("*.pdf")
        if path.name.startswith("Weekly ")
    )

    discovery = discover_pdf_tables_automatically(electricity_pdf.read_bytes())

    assert discovery.searched_pages == tuple(range(1, 7))
    assert len(discovery.candidates) == 5
    assert discovery.logical_table_count == 7
    assert discovery.pages_without_candidates == (1,)
    assert "3 logical sections" in discovery.candidates[0].label


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


def test_generic_cleanup_reconstructs_bilingual_multirow_header():
    electricity_pdf = next(
        path
        for path in Path("sample_documents").glob("*.pdf")
        if path.name.startswith("Weekly ")
    )
    raw = discover_table_candidates(electricity_pdf, (5,))[0].dataframe

    suggested = suggest_header_rows(raw)
    cleaned = clean_generic_table(
        raw,
        header_row=suggested[0],
        header_row_count=len(suggested),
        remove_devanagari=True,
    )

    assert suggested == (2, 3)
    assert cleaned.shape == (8, 10)
    assert cleaned.columns[:4].tolist() == [
        "Date",
        "BHUTAN | Energy Exchange (In MU)",
        "BHUTAN | Day Peak (MW)",
        "BHUTAN | Day Average (MW)",
    ]
    assert not any(
        "\u0900" <= character <= "\u097f"
        for column in cleaned.columns
        for character in column
    )


def test_generic_extractor_splits_vertically_stacked_page_tables():
    electricity_pdf = next(
        path
        for path in Path("sample_documents").glob("*.pdf")
        if path.name.startswith("Weekly ")
    )
    combined = discover_table_candidates(electricity_pdf, (2,))[0].dataframe

    sections = split_table_sections(combined)

    assert [(section.start_row, section.end_row) for section in sections] == [
        (0, 10),
        (10, 20),
        (20, 29),
    ]
    assert [section.dataframe.shape[0] for section in sections] == [10, 10, 9]
    assert "Evening Demand Met" in sections[0].title
    assert "Energy Met & Hydro Generation" in sections[1].title
    assert "All India Grid Frequency" in sections[2].title

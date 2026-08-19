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
    find_suspicious_columns,
    remove_repeated_header_rows,
    split_table_sections,
    suggest_header_rows,
)
from app.extraction.table_extractor import normalize_table_span_columns


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


def test_generic_cleanup_removes_repeated_continuation_headers():
    raw = pd.DataFrame(
        [
            ["State", "Cases", "Deaths"],
            ["Assam", "10", "1"],
            ["State", "Cases", "Deaths"],
            ["Bihar", "20", "2"],
        ]
    )

    cleaned = clean_generic_table(raw, header_row=0)

    assert cleaned.to_dict(orient="records") == [
        {"State": "Assam", "Cases": "10", "Deaths": "1"},
        {"State": "Bihar", "Cases": "20", "Deaths": "2"},
    ]


def test_generic_cleanup_merges_wrapped_narrative_continuation_rows():
    raw = pd.DataFrame(
        [
            ["Unique ID", "State", "Cases", "Comments/Action Taken"],
            ["A-1", "Assam", "12", "Cases were reported from Village"],
            [None, None, None, "Namsai. Cases presented with fever."],
            [None, None, None, "District RRT investigated the outbreak."],
            ["B-1", "Bihar", "5", "Separate record."],
        ]
    )

    cleaned = clean_generic_table(raw, header_row=0)
    unmerged = clean_generic_table(
        raw,
        header_row=0,
        merge_continuation_rows=False,
    )

    assert len(cleaned) == 2
    assert cleaned.loc[0, "Comments/Action Taken"] == (
        "Cases were reported from Village Namsai. Cases presented with "
        "fever. District RRT investigated the outbreak."
    )
    assert len(unmerged) == 4


def test_suspicious_columns_flags_populated_placeholder_and_short_neighbor():
    table = pd.DataFrame(
        {"Date": ["2 May"], "D D": ["1"], "Column_9": ["8"]}
    )

    assert find_suspicious_columns(table) == ("D D", "Column_9")
    assert remove_repeated_header_rows(table).equals(table)


def test_span_normalization_collapses_header_only_split_columns():
    first_page = pd.DataFrame(
        [
            ["ID", "State", "Cases", "", "Date of", "", "Status"],
            [None, None, None, None, "Start of", None, None],
            [None, None, None, None, "Outbreak", None, None],
            ["A-1", "Assam", "08", "19-05-18", None, None, "Open"],
        ]
    )
    continuation = pd.DataFrame(
        [["B-1", "Bihar", "24", "18-05-18", "Closed"]]
    )

    normalized = normalize_table_span_columns((first_page, continuation))
    combined = pd.concat(normalized, ignore_index=True)

    assert [table.shape[1] for table in normalized] == [5, 5]
    assert combined.iloc[0].tolist() == [
        "ID",
        "State",
        "Cases",
        "Date of",
        "Status",
    ]
    assert combined.iloc[1, 3] == "Start of"
    assert combined.iloc[2, 3] == "Outbreak"
    assert combined.iloc[-1].tolist() == [
        "B-1",
        "Bihar",
        "24",
        "18-05-18",
        "Closed",
    ]
    assert suggest_header_rows(combined) == (0, 1, 2)


def test_cleanup_moves_detached_header_onto_its_data_column():
    raw = pd.DataFrame(
        [
            ["ID", "State", "Cases", "", "Date of", "", "Status"],
            [None, None, None, None, "Start of", None, None],
            [None, None, None, None, "Outbreak", None, None],
            ["A-1", "Assam", "08", "19-05-18", None, None, "Open"],
        ]
    )

    suggested = suggest_header_rows(raw)
    cleaned = clean_generic_table(
        raw,
        header_row=suggested[0],
        header_row_count=len(suggested),
    )

    assert suggested == (0, 1, 2)
    assert cleaned.columns.tolist() == [
        "ID",
        "State",
        "Cases",
        "Date of | Start of | Outbreak",
        "Status",
    ]
    assert cleaned.iloc[0].tolist() == [
        "A-1",
        "Assam",
        "08",
        "19-05-18",
        "Open",
    ]


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

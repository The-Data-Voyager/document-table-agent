from pathlib import Path
from collections.abc import Iterable

import pandas as pd
import pdfplumber


def _validate_pdf_path(pdf_path):
    """
    Validate that the supplied path exists and is a PDF.

    Parameters
    ----------
    pdf_path : str or Path
        Path to the PDF document.

    Returns
    -------
    Path
        Validated Path object.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file, got: {pdf_path.suffix}"
        )

    return pdf_path


def extract_tables_from_page(
    pdf_path,
    page_number,
    table_settings=None
):
    """
    Extract all tables detected on one PDF page.

    Page numbers use human numbering:
    page 1, page 2, page 3...

    Parameters
    ----------
    pdf_path : str or Path
        Path to the PDF.

    page_number : int
        Human-readable PDF page number.

    table_settings : dict, optional
        Custom pdfplumber table extraction settings.

    Returns
    -------
    list
        Raw tables detected by pdfplumber.
    """

    pdf_path = _validate_pdf_path(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:

        total_pages = len(pdf.pages)

        if page_number < 1 or page_number > total_pages:
            raise ValueError(
                f"Page number must be between 1 and "
                f"{total_pages}"
            )

        page = pdf.pages[page_number - 1]

        if table_settings is None:
            tables = page.extract_tables()
        else:
            tables = page.extract_tables(
                table_settings=table_settings
            )

    return tables


def table_to_dataframe(table):
    """
    Convert one raw extracted table into a Pandas DataFrame.

    The function intentionally performs no cleaning,
    header correction or transformation.

    Parameters
    ----------
    table : list
        Raw table returned by pdfplumber.

    Returns
    -------
    pandas.DataFrame
        Raw table represented as a DataFrame.
    """

    if table is None:
        raise ValueError("Table cannot be None.")

    if len(table) == 0:
        raise ValueError("Table is empty.")

    return pd.DataFrame(table)


def extract_table_as_dataframe(
    pdf_path,
    page_number,
    table_index=0,
    table_settings=None
):
    """
    Extract one table from a PDF page and return it
    as a raw Pandas DataFrame.

    Parameters
    ----------
    pdf_path : str or Path
        Path to the PDF.

    page_number : int
        Human-readable page number.

    table_index : int, default=0
        Zero-based index of the detected table.

    table_settings : dict, optional
        Custom pdfplumber table settings.

    Returns
    -------
    pandas.DataFrame
        Raw extracted table.
    """

    tables = extract_tables_from_page(
        pdf_path=pdf_path,
        page_number=page_number,
        table_settings=table_settings
    )

    if not tables:
        raise ValueError(
            f"No tables detected on page {page_number}."
        )

    if table_index < 0 or table_index >= len(tables):
        raise IndexError(
            f"table_index must be between 0 and "
            f"{len(tables) - 1}."
        )

    selected_table = tables[table_index]

    return table_to_dataframe(selected_table)


def extract_table_span_as_dataframe(
    pdf_path,
    page_numbers: Iterable[int],
    table_index=0,
    table_settings=None,
):
    """Extract the same table index across ordered pages and concatenate it.

    The returned DataFrame is still raw extraction output. Different page
    widths are preserved by Pandas as additional positional columns; no header
    correction, row merging, or content cleaning occurs here.
    """

    pdf_path = _validate_pdf_path(pdf_path)
    pages = tuple(page_numbers)
    if not pages:
        raise ValueError("page_numbers cannot be empty.")
    if any(isinstance(page, bool) or not isinstance(page, int) for page in pages):
        raise TypeError("page_numbers must contain only integers.")
    if any(page < 1 for page in pages):
        raise ValueError("page_numbers must use positive human page numbers.")
    if pages != tuple(sorted(pages)) or len(pages) != len(set(pages)):
        raise ValueError("page_numbers must be ascending and unique.")
    if isinstance(table_index, bool) or not isinstance(table_index, int):
        raise TypeError("table_index must be an integer.")
    if table_index < 0:
        raise ValueError("table_index cannot be negative.")

    frames = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for page_number in pages:
            if page_number > total_pages:
                raise ValueError(
                    f"Page number must be between 1 and {total_pages}."
                )
            page = pdf.pages[page_number - 1]
            tables = (
                page.extract_tables()
                if table_settings is None
                else page.extract_tables(table_settings=table_settings)
            )
            if not tables:
                raise ValueError(
                    f"No tables detected on page {page_number}."
                )
            if table_index >= len(tables):
                raise IndexError(
                    f"table_index {table_index} is unavailable on page "
                    f"{page_number}; detected {len(tables)} tables."
                )
            frames.append(table_to_dataframe(tables[table_index]))

    return pd.concat(frames, ignore_index=True, sort=False)


def get_table_shape(table):
    """
    Return row and column counts for a raw table.

    Returns
    -------
    dict
        Dictionary containing rows and columns.
    """

    if table is None or len(table) == 0:
        return {
            "rows": 0,
            "columns": 0
        }

    return {
        "rows": len(table),
        "columns": max(
            len(row) if row is not None else 0
            for row in table
        )
    }

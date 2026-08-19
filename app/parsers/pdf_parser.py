
from pathlib import Path
import fitz


def open_pdf(pdf_path):
    """
    Open and validate a PDF file.

    Parameters
    ----------
    pdf_path : str or Path
        Path to the PDF file.

    Returns
    -------
    fitz.Document
        Opened PyMuPDF document.
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

    return fitz.open(pdf_path)


def get_page_count(pdf_path):
    """
    Return the total number of pages in a PDF.
    """

    with open_pdf(pdf_path) as doc:
        return len(doc)


def extract_page_text(pdf_path, page_number):
    """
    Extract text from one PDF page.

    Page numbers use human numbering:
    page 1, page 2, page 3...
    """

    with open_pdf(pdf_path) as doc:

        if page_number < 1 or page_number > len(doc):
            raise ValueError(
                f"Page number must be between 1 and {len(doc)}"
            )

        page = doc[page_number - 1]

        return page.get_text()


def search_pdf(pdf_path, search_text):
    """
    Search for text across the PDF.

    Returns a list of human-readable page numbers.
    """

    if not search_text.strip():
        raise ValueError("Search text cannot be empty.")

    matching_pages = []

    with open_pdf(pdf_path) as doc:

        for page_index, page in enumerate(doc):

            text = page.get_text()

            if search_text.lower() in text.lower():
                matching_pages.append(page_index + 1)

    return matching_pages


def pdf_has_text(pdf_path, minimum_characters=50):
    """
    Check whether a PDF contains extractable text.

    This will later help us decide whether OCR
    may be required.
    """

    with open_pdf(pdf_path) as doc:

        total_characters = 0

        for page in doc:

            text = page.get_text().strip()

            total_characters += len(text)

            if total_characters >= minimum_characters:
                return True

    return False

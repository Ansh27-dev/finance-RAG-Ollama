from pathlib import Path
from typing import Any

import pdfplumber


def extract_pdf_pages(pdf_path: Path) -> list[dict[str, Any]]:
    """Extract selectable text from each page while preserving source metadata."""
    pages: list[dict[str, Any]] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            cleaned = " ".join(text.split())
            if cleaned:
                pages.append(
                    {
                        "text": cleaned,
                        "file_name": pdf_path.name,
                        "page_number": index,
                        "quarter": infer_quarter_from_filename(pdf_path.name),
                    }
                )

    return pages


def infer_quarter_from_filename(file_name: str) -> str:
    stem = Path(file_name).stem.replace("-", "_").replace(" ", "_")
    parts = [part for part in stem.split("_") if part]
    quarter_parts = [
        part
        for part in parts
        if part.upper().startswith("Q") or part.upper().startswith("FY")
    ]
    return " ".join(quarter_parts) if quarter_parts else stem

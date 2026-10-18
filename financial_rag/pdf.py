"""Context: local PDF ingestion preserves page-level financial context.

PDF content is evidence, never instructions to execute. No OCR, external document
uploads, or model calls happen here. Optional PyMuPDF is loaded only for PDFs.
"""

import hashlib
from html import escape
from pathlib import Path
import re

from .core import Document, Section, stable_id


def clean_page_text(text: str) -> str:
    """Context: remove identifiable browser print furniture, preserving table spacing.

    We do not collapse horizontal whitespace: it provides essential financial
    column alignment. Currency, parentheses, report footers and footnotes remain.
    """
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^\d{1,2}/\d{1,2}/\d{2,4},\s+\d{1,2}:\d{2}\s+[AP]M\b", stripped):
            continue
        if re.match(r"^https?://www\.sec\.gov/Archives/.*\s+\d+/\d+\s*$", stripped):
            continue
        lines.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def page_location(number: int, text: str) -> str:
    """Context: physical PDF pages and printed filing pages are different identifiers."""
    printed = re.search(r"\bForm\s+10-[KQ]\s*\|\s*(\d+)\b", text, re.I)
    return f"PDF Page {number}" + (f" (printed page {printed[1]})" if printed else "")


def section_heading(text: str, previous: str) -> str:
    """Context: recognize explicit statement/item headings without guessing missing ones."""
    candidates = [line.strip() for line in text.splitlines() if line.strip()]
    for line in candidates:
        if re.match(r"^CONSOLIDATED (STATEMENTS|BALANCE SHEETS)", line):
            return line
    for line in candidates[:18]:
        if re.match(r"^(Item\s+\d+[A-Z]?\.|Note\s+\d+\s*[–—-])", line):
            return line
    return previous


def table_html(table) -> str:
    """Context: keep raw cell grids and external headers as supplementary evidence.

    Merged cells can produce apparent column shifts in flattened grids. The full
    page layout remains primary, and this warning travels with each table.
    """
    rows = table.extract()
    header = table.header.names if table.header and table.header.external else []
    if header:
        rows = [header] + rows
    content = []
    for row in rows:
        cells = " | ".join(escape(re.sub(r"\s+", " ", str(cell or "")).strip()) for cell in row)
        content.append(cells)
    return ("Supplementary detected table; merged cells may shift columns. "
            "Use the preserved page layout to verify year/value alignment.\n"
            "<table>\n" + "\n".join(content) + "\n</table>")


def parse_pdf(path: Path, ticker: str, year: int, quarter="FY", doc_type="10-K",
              source="", accession="", filing_date="") -> Document:
    """Context: one canonical section per page keeps financial headers with values.

    Reject unreadable/image-only pages rather than silently returning an incomplete
    filing. A page with no text but no images is treated as an explicit blank page.
    """
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("Install PDF support: uv sync --locked --extra pdf") from exc
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    doc_id = f"{ticker.upper()}_{year}_{quarter}_{doc_type}_PDF_{digest[:8]}"
    sections, warnings, blank_pages = [], [], []
    table_count = 0
    heading = "Document"
    with pymupdf.open(path) as pdf:
        if pdf.needs_pass:
            raise ValueError("Password-protected PDF: supply an unlocked local copy.")
        total_pages = len(pdf)
        for number, page in enumerate(pdf, 1):
            text = clean_page_text(page.get_text("text", sort=True))
            if not text.strip():
                if page.get_images():
                    raise ValueError(f"PDF page {number} has images but no readable text; OCR is required.")
                blank_pages.append(number)
                continue
            heading = section_heading(text, heading)
            location = page_location(number, text)
            supplements = []
            # Context: ruled-table detection is supplemental, not a completeness guarantee.
            # If detection fails, preserve the original page text and flag the page.
            try:
                finder = page.find_tables()
                for table in finder.tables:
                    supplements.append(table_html(table))
                table_count += len(finder.tables)
            except Exception as exc:
                warnings.append({"page": number, "issue": "table-detection-failed", "type": type(exc).__name__})
            content = text
            if supplements:
                content += "\n\n" + "\n\n".join(supplements)
            sections.append(Section(stable_id(doc_id, str(number)), content, heading, location))
    if not sections:
        raise ValueError("PDF contains no readable text; OCR is required.")
    return Document(doc_id, ticker.upper(), year, quarter, doc_type,
                    source or str(path.resolve()), digest, sections, accession, filing_date,
                    provenance={"format": "pdf", "parser": "pymupdf", "parser_version": pymupdf.VersionBind,
                                "pages": total_pages, "detected_tables": table_count,
                                "blank_pages": blank_pages, "warnings": warnings,
                                "local_artifact": str(path.resolve()),
                                "layout_policy": "page-layout-primary; detected-cell-grids-supplementary"})

"""Context: parse primary HTML/text or text-based PDFs with source locations.

Ambiguous multipart submissions are rejected. PDF pages retain their physical
page numbers; printed report numbers are added only when found in visible text.
"""

from dataclasses import asdict
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .core import Document, Section, stable_id


class FilingParser(HTMLParser):
    """Context: collect block text and table rows while ignoring executable markup."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str, str]] = []
        self.buffer: list[str] = []
        self.skip = 0
        self.hidden = 0
        self.table_depth = 0
        self.heading = ""
        self.anchor = ""
        self.row: list[str] = []
        self.cell: list[str] | None = None
        self.rows: list[list[str]] = []

    def flush(self):
        text = re.sub(r"\s+", " ", " ".join(self.buffer)).strip()
        if text:
            self.blocks.append((self.heading, self.anchor, text))
        self.buffer = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "ix:hidden"}:
            self.skip += 1
        if self.skip:
            return
        attrs = dict(attrs)
        if tag == "table":
            if self.table_depth == 0:
                self.flush()
                self.rows = []
            self.table_depth += 1
        elif self.table_depth:
            if tag == "tr":
                self.row = []
            elif tag in {"td", "th"}:
                self.cell = []
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.flush()
            self.heading = tag
            self.anchor = attrs.get("id", self.anchor)
        elif tag in {"p", "div", "li", "br"}:
            self.flush()
            self.anchor = attrs.get("id", self.anchor)

    def handle_data(self, data):
        if self.skip:
            return
        if self.table_depth:
            if self.cell is not None:
                self.cell.append(data)
        else:
            self.buffer.append(data)

    def handle_endtag(self, tag):
        if tag in {"script", "style", "ix:hidden"}:
            self.skip = max(0, self.skip - 1)
            return
        if self.skip:
            return
        if self.table_depth:
            if tag in {"td", "th"} and self.cell is not None:
                self.row.append(re.sub(r"\s+", " ", " ".join(self.cell)).strip())
                self.cell = None
            elif tag == "tr" and self.row:
                self.rows.append(self.row)
                self.row = []
            elif tag == "table":
                self.table_depth -= 1
                if self.table_depth == 0:
                    table = "\n".join(" | ".join(row) for row in self.rows)
                    self.blocks.append(("table", self.anchor, "<table>\n" + table + "\n</table>"))
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.flush()
            self.heading = ""
        elif tag in {"p", "div", "li"}:
            self.flush()


def parse_document(path: Path, ticker: str, year: int, quarter="FY", doc_type="10-K",
                   source="", accession="", filing_date="") -> Document:
    """Context: retain original bytes' identity and reject unsupported formats."""
    if path.suffix.lower() == ".pdf":
        from .pdf import parse_pdf
        return parse_pdf(path, ticker, year, quarter, doc_type, source, accession, filing_date)
    if path.suffix.lower() not in {".html", ".htm", ".txt", ".md"}:
        raise ValueError("Use primary HTML, UTF-8 text, Markdown, or a text-based PDF.")
    raw = path.read_text(encoding="utf-8-sig")
    if re.search(r"<DOCUMENT>", raw, re.I):
        raise ValueError("Multipart SEC submission: supply the primary filing HTML instead.")
    source = source or str(path.resolve())
    digest = stable_id(raw)
    doc_id = f"{ticker.upper()}_{year}_{quarter}_{doc_type}_{digest[:8]}"
    if re.search(r"<(html|body|table|div|p|h[1-6])\b", raw, re.I):
        parser = FilingParser()
        parser.feed(raw)
        parser.flush()
        blocks = parser.blocks
    else:
        blocks = [("heading" if line.startswith("#") else "", "", line.strip("# "))
                  for line in re.split(r"\n\s*\n", raw) if line.strip()]
    sections = []
    heading = "Document"
    for number, (kind, anchor, text) in enumerate(blocks):
        if not text.strip():
            continue
        if kind.startswith("h"):
            heading = text
        location = f"Anchor {anchor}" if anchor else f"Block {number + 1}"
        sections.append(Section(stable_id(doc_id, str(number)), text, heading, location))
    if not sections:
        raise ValueError("No readable content found.")
    return Document(doc_id, ticker.upper(), year, quarter, doc_type, source, digest,
                    sections, accession, filing_date)


def save_documents(documents, path: Path):
    """Context: originals stay untouched; canonical output is a separate artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(doc) for doc in documents], indent=2), encoding="utf-8")


def download_sec(url: str, destination: Path, user_agent: str):
    """Context: fetch one explicitly selected SEC primary document, never scrape blindly."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {"www.sec.gov", "www.sec.gov."}:
        raise ValueError("Expected an https://www.sec.gov/Archives/ primary filing URL.")
    if not parsed.path.startswith("/Archives/"):
        raise ValueError("Only SEC Archives primary document URLs are accepted.")
    if "@" not in user_agent or "example.com" in user_agent:
        raise ValueError("Set SEC_USER_AGENT to your organization and real contact email.")
    request = Request(url, headers={"User-Agent": user_agent, "Accept-Encoding": "identity"})
    with urlopen(request, timeout=60) as response:
        if urlparse(response.url).hostname != "www.sec.gov":
            raise ValueError("Unexpected redirect outside SEC host.")
        content = response.read(30_000_001)
    if len(content) > 30_000_000:
        raise ValueError("Document exceeds the 30 MB download limit.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)

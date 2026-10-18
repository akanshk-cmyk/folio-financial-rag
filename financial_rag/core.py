"""Context: shared records and local persistence keep every result traceable.

These record types use only the standard library; pipeline orchestration and
chunking separately require the project's core LangChain dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re


REFUSAL = "Information not available in provided financial disclosures."


def stable_id(*parts: str) -> str:
    """Context: content-derived IDs make repeated ingestion idempotent."""
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:24]


def tokens(text: str) -> list[str]:
    """Context: lexical terms preserve numeric strings and financial abbreviations."""
    return re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", text.lower())


@dataclass
class Section:
    """Context: one source section retains headings and a stable location."""
    id: str
    text: str
    section_path: str
    location: str


@dataclass
class Document:
    """Context: fiscal period is explicit input, never inferred from a filename."""
    id: str
    ticker: str
    fiscal_year: int
    fiscal_quarter: str
    doc_type: str
    source: str
    content_hash: str
    sections: list[Section]
    accession: str = ""
    filing_date: str = ""
    synthetic: bool = False
    provenance: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """Context: children carry source metadata and point to a stored parent."""
    id: str
    parent_id: str
    doc_id: str
    text: str
    ticker: str
    fiscal_year: int
    fiscal_quarter: str
    doc_type: str
    section_path: str
    location: str
    source: str
    synthetic: bool = False

    def citation(self) -> str:
        return f"[{self.doc_id}, {self.section_path}, {self.location}]"


@dataclass
class Index:
    """Context: JSON snapshots include configuration, parents, and dense vectors."""
    manifest: dict
    chunks: list[Chunk]
    parents: dict[str, str]
    vectors: list[list[float]] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> Index:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["chunks"] = [Chunk(**item) for item in data["chunks"]]
        return cls(**data)


def load_documents(path: Path) -> list[Document]:
    """Context: canonical JSON separates parsing from repeated experiments."""
    result = []
    for item in json.loads(path.read_text(encoding="utf-8")):
        item["sections"] = [Section(**section) for section in item["sections"]]
        result.append(Document(**item))
    return result

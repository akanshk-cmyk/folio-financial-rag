"""Context: smoke checks for the user-supplied Apple FY2025 PDF.

Reference rows were visually checked on PDF page 41 / printed page 29. This
verifies extraction and lexical retrieval, not all disclosures or LLM accuracy.
Run from the project root using the project virtual environment.
"""

import json
from pathlib import Path
import re

from financial_rag.core import Index, load_documents
from financial_rag.generation import answer
from financial_rag.retrieval import retrieve


def main():
    """Context: fail if financial rows or their year-ordered values are lost."""
    document = next(doc for doc in load_documents(Path("data/documents.json"))
                    if doc.accession == "0000320193-25-000079" and doc.provenance.get("format") == "pdf")
    statement = next(section for section in document.sections
                     if section.location == "PDF Page 41 (printed page 29)")
    primary = statement.text.split("Supplementary detected table;")[0]
    references = {"Total net sales": ["416,161", "391,035", "383,285"],
                  "Operating income": ["133,050", "123,216", "114,301"],
                  "Net income": ["112,010", "93,736", "96,995"]}
    checks = []
    for label, values in references.items():
        row = next((line for line in primary.splitlines()
                    if re.match(r"^\s*" + re.escape(label) + r"\s+", line)), "")
        extracted = re.findall(r"\d[\d,]*", row)
        checks.append({"metric": label, "years": [2025, 2024, 2023], "unit": "USD millions",
                       "expected": values, "extracted": extracted, "passed": extracted == values,
                       "citation": f"[{document.id}, {statement.section_path}, {statement.location}]"})
    if not all(check["passed"] for check in checks):
        raise ValueError("Statement extraction changed; inspect the original PDF before proceeding.")
    index = Index.load(Path("data/apple-2025-lexical.json"))
    hits, pool = retrieve(index, "total net sales", filters={"ticker": "AAPL", "fiscal_year": 2025}, expand=True)
    if not hits or hits[0].chunk.location != statement.location:
        raise ValueError("Revenue retrieval did not rank the verified statement first.")
    result = answer("What evidence supports Apple's FY2025 total net sales?", hits)
    Path("runs").mkdir(exist_ok=True)
    Path("runs/apple-2025-revenue-evidence.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = {"document_id": document.id, "source": document.source, "provenance": document.provenance,
              "canonical_sections": len(document.sections), "index_chunks": len(index.chunks),
              "checks": checks, "retrieval_first_result": hits[0].chunk.location,
              "candidate_pool": pool, "scope": "Extraction and lexical retrieval only; no LLM evaluation."}
    Path("runs/apple-2025-ingestion-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Verified {len(checks)} statement rows and revenue retrieval; report: runs/apple-2025-ingestion-report.json")


if __name__ == "__main__":
    main()

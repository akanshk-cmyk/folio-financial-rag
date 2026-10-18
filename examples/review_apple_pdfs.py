"""Context: verify both locally supplied Apple reports before model experiments.

Reference rows were visually checked on PDF page 41 / printed page 29 of each
file. This verifies extraction, filing-year filtering, and an overlapping-period
consistency check. It is not an evaluation of language-model answer quality.
"""

import json
from pathlib import Path
import re

from financial_rag.core import Index, load_documents
from financial_rag.generation import answer
from financial_rag.retrieval import retrieve


REFERENCES = {
    2024: {"accession": "0000320193-24-000123", "years": [2024, 2023, 2022],
           "rows": {"Total net sales": ["391,035", "383,285", "394,328"],
                    "Operating income": ["123,216", "114,301", "119,437"],
                    "Net income": ["93,736", "96,995", "99,803"]}},
    2025: {"accession": "0000320193-25-000079", "years": [2025, 2024, 2023],
           "rows": {"Total net sales": ["416,161", "391,035", "383,285"],
                    "Operating income": ["133,050", "123,216", "114,301"],
                    "Net income": ["112,010", "93,736", "96,995"]}},
}


def main():
    """Context: produce one audit report and separate cited evidence per filing year."""
    documents = load_documents(Path("data/documents.json"))
    index = Index.load(Path("data/apple-2024-2025-lexical.json"))
    reports = []
    for year, reference in REFERENCES.items():
        document = next(doc for doc in documents if doc.accession == reference["accession"]
                        and doc.provenance.get("format") == "pdf")
        statement = next(section for section in document.sections
                         if section.location == "PDF Page 41 (printed page 29)")
        primary = statement.text.split("Supplementary detected table;")[0]
        checks = []
        for label, expected in reference["rows"].items():
            row = next((line for line in primary.splitlines()
                        if re.match(r"^\s*" + re.escape(label) + r"\s+", line)), "")
            values = re.findall(r"\d[\d,]*", row)
            checks.append({"metric": label, "years": reference["years"], "unit": "USD millions",
                           "expected": expected, "extracted": values, "passed": values == expected})
        if not all(check["passed"] for check in checks):
            raise ValueError(f"FY{year} extraction changed; review the original PDF.")
        hits, pool = retrieve(index, "total net sales", filters={"ticker": "AAPL", "fiscal_year": year}, expand=True)
        statement_rank = next((rank for rank, hit in enumerate(hits, 1)
                               if hit.chunk.location == statement.location and hit.chunk.doc_id == document.id), None)
        if statement_rank is None:
            raise ValueError(f"FY{year} income statement was absent from the top three evidence passages.")
        if any(hit.chunk.fiscal_year != year for hit in hits):
            raise ValueError("Cross-year filter leakage detected.")
        result = answer(f"What evidence supports Apple's FY{year} total net sales?", hits)
        Path("runs").mkdir(exist_ok=True)
        Path(f"runs/apple-{year}-revenue-evidence.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        report = {"fiscal_year": year, "document_id": document.id, "source": document.source,
                  "provenance": document.provenance, "canonical_sections": len(document.sections),
                  "checks": checks, "retrieval_first_result": hits[0].chunk.location,
                  "verified_statement_rank": statement_rank,
                  "candidate_pool": pool, "filing_year_filter_passed": True}
        reports.append(report)
        Path(f"runs/apple-{year}-ingestion-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Context: the FY2024 column in the 2025 filing must match the FY2024 filing's
    # current-year column for these three reviewed metrics. This is a targeted
    # consistency check, not automatic restatement handling for the whole corpus.
    consistency = []
    for earlier, later in zip(reports[0]["checks"], reports[1]["checks"]):
        passed = earlier["extracted"][0] == later["extracted"][1]
        consistency.append({"metric": earlier["metric"], "period": "FY2024", "passed": passed})
    if not all(item["passed"] for item in consistency):
        raise ValueError("Reviewed FY2024 values differ across reports; investigate restatement/version context.")
    combined = {"documents": reports, "total_index_chunks": len(index.chunks),
                "overlapping_period_consistency": consistency,
                "scope": "Extraction and local BM25 retrieval; no generated-answer evaluation."}
    output = Path("runs/apple-2024-2025-ingestion-report.json")
    output.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"Both filings checked: 6 statement rows, 2 filtered searches, 3 overlap checks. Report: {output}")


if __name__ == "__main__":
    main()

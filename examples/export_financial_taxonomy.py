"""Context: export the app's actual retrieval vocabulary for spreadsheet review.

Run from the project root: uv run python examples/export_financial_taxonomy.py
No API keys, model downloads, or external requests are needed.
"""

import csv
from pathlib import Path

from financial_rag.taxonomy import CONCEPTS, TAXONOMY_VERSION


def main():
    """Context: keep the reviewable CSV synchronized with the Python source of truth."""
    target = Path(__file__).resolve().parents[1] / "data" / "financial_taxonomy.csv"
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("Version", "Concept", "Category", "Question aliases",
                         "Candidate row labels", "Value type", "Validation caution"))
        for item in CONCEPTS:
            writer.writerow((TAXONOMY_VERSION, item.key, item.category,
                             " | ".join(item.aliases), " | ".join(item.rows),
                             item.kind, item.caution))
    print(f"Exported {len(CONCEPTS)} concepts to {target}")


if __name__ == "__main__":
    main()

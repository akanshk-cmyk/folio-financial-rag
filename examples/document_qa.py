"""Context: run the reusable DocumentQA class against a saved PDF vector snapshot.

Example: uv run --no-sync python examples/document_qa.py --store data/document-qa \
  --url https://issuer.example/report.pdf --pdf report.pdf --ticker TEST --year 2025 \
  --question "What was total revenue?"
Omit --pdf/--ticker/--year on later runs to reuse stored embeddings.
"""

import argparse
import json

from financial_rag.document_qa import DocumentQA


def main():
    """Context: explicit flags distinguish first-time PDF indexing from stored-vector QA."""
    parser=argparse.ArgumentParser(description="Ask one indexed financial PDF with cited evidence.")
    parser.add_argument("--store", required=True, help="Local vector snapshot JSON or directory")
    parser.add_argument("--url", required=True, help="Exact source URL recorded for the PDF")
    parser.add_argument("--question", required=True)
    parser.add_argument("--pdf", help="Downloaded PDF to index on first use; no URL download is performed")
    parser.add_argument("--ticker")
    parser.add_argument("--year", type=int)
    parser.add_argument("--provider", choices=("nebius","groq","openai","extractive"), default="nebius")
    parser.add_argument("--model")
    parser.add_argument("--revision", help="Embedding revision recorded in an existing snapshot")
    args=parser.parse_args()
    if args.pdf and (not args.ticker or args.year is None):
        parser.error("--pdf requires --ticker and --year")
    try:
        qa=DocumentQA(args.store, provider=args.provider, model=args.model, revision=args.revision)
        if args.pdf:
            qa.index_pdf(args.pdf, source_url=args.url, ticker=args.ticker, fiscal_year=args.year)
        else:
            qa.load_document(args.url)
        result=qa.ask(args.question)
    except (ValueError,RuntimeError,OSError,ImportError) as exc:
        parser.exit(1,f"Error: {exc}\n")
    print(result["answer"])
    print("\nSupporting evidence:")
    print(json.dumps(result["supporting_evidence"],indent=2))


if __name__ == "__main__":
    main()

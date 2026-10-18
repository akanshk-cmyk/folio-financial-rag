"""Context: explicit subcommands separate local work from network/API actions.

Run `python -m financial_rag --help` for the documented entry point.
"""

import argparse
import json
import os
from pathlib import Path
import sys

from .chunking import build_index
from .chains import build_rag_chain
from .core import Index, load_documents
from .generation import answer, load_environment
from .ingest import download_sec, parse_document, save_documents
from .retrieval import retrieve


def write_json(path, data):
    """Context: command outputs are human-readable artifacts for later review."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_embedder(model=None, revision=None):
    """Context: model loading is deferred so offline commands never download weights."""
    from .embedding import Embedder, MODEL
    return Embedder(model or MODEL, revision)


def demo(output):
    """Context: fixtures are fictional and clearly labeled; no Apple figures are fabricated."""
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    documents = []
    for year, revenue, income in ((2024, "100", "20"), (2025, "120", "30")):
        path = folder / f"SYNTHETIC_DEMO_{year}.html"
        path.write_text(f"""<html><body><h1 id="results">SYNTHETIC DEMO — not a real filing</h1>
<p>Fictional DEMO company FY{year}. Currency: USD; values in millions; consolidated GAAP.</p>
<table><tr><th>Metric</th><th>FY{year}</th></tr>
<tr><td>Revenue</td><td>{revenue}</td></tr>
<tr><td>Operating income</td><td>{income}</td></tr></table>
<p id="note1">Footnote 1: Fictional figures are for software testing only.</p>
<h2 id="risk">Liquidity risks</h2><p>Customer concentration may affect cash collections.</p>
</body></html>""", encoding="utf-8")
        doc = parse_document(path, "DEMO", year, doc_type="Synthetic")
        doc.synthetic = True
        documents.append(doc)
    save_documents(documents, folder / "documents.json")
    index = build_index(documents, "enhanced")
    index.save(folder / "index.json")
    hits, pool = retrieve(index, "Revenue", filters={"ticker": "DEMO", "fiscal_year": 2025}, expand=True)
    result = answer("What revenue evidence is available for DEMO FY2025?", hits)
    write_json(folder / "answer.json", result)
    print(result["answer"])
    print(f"\nOffline demo artifacts: {folder.resolve()}")


def parser():
    """Context: defaults are local and non-billable; hosted generation is opt-in."""
    root = argparse.ArgumentParser(description="Financial RAG: documented baseline and enhanced pipelines")
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser("demo", help="Run fictional data through local parsing and BM25 search")
    p.add_argument("--output", default="runs/demo")
    p = commands.add_parser("ingest", help="Append a primary HTML/text/PDF filing to canonical JSON")
    p.add_argument("file", type=Path)
    p.add_argument("--ticker", required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--quarter", default="FY")
    p.add_argument("--doc-type", default="10-K")
    p.add_argument("--source", default="")
    p.add_argument("--accession", default="")
    p.add_argument("--filing-date", default="")
    p.add_argument("--output", type=Path, default=Path("data/documents.json"))
    p = commands.add_parser("download", help="Download one explicit SEC Archives primary document")
    p.add_argument("url")
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("index", help="Create a repeatable local search snapshot")
    p.add_argument("--documents", type=Path, default=Path("data/documents.json"))
    p.add_argument("--strategy", choices=["fixed", "semantic", "enhanced"], required=True)
    p.add_argument("--backend", choices=["dense", "lexical"], default="dense")
    p.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--revision", help="Pin a Hugging Face commit for reproducible embeddings")
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("ask", help="Retrieve cited evidence or explicitly request hosted generation")
    p.add_argument("question")
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--retrieval", choices=["local", "pinecone"], default="local")
    p.add_argument("--pinecone-index", help="Overrides PINECONE_INDEX")
    p.add_argument("--namespace", help="Required for Pinecone search")
    p.add_argument("--ticker")
    p.add_argument("--year", type=int, action="append", help="Repeat for multi-year comparisons")
    p.add_argument("--quarter")
    p.add_argument("--doc-type")
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--expand", action="store_true")
    p.add_argument("--candidates", type=int, default=10)
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--provider", choices=["extractive", "groq", "openai", "nebius"], default="extractive")
    p.add_argument("--model")
    p.add_argument("--output", type=Path, default=Path("runs/latest-answer.json"))
    p = commands.add_parser("experiment", help="Run controlled dense A/B/C comparisons")
    p.add_argument("--fixed", type=Path, required=True)
    p.add_argument("--semantic", type=Path, required=True)
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--retrieval", choices=["local", "pinecone"], default="local")
    p.add_argument("--pinecone-index", help="Overrides PINECONE_INDEX; uses fixed/semantic namespaces")
    p.add_argument("--provider", choices=["extractive", "groq", "openai", "nebius"], default="extractive")
    p.add_argument("--model", help="Overrides the selected provider model setting")
    p.add_argument("--output", type=Path, default=Path("runs/experiment.json"))
    p.add_argument("--resume",action="store_true",help="Resume identical settings, retaining completed answers")
    p = commands.add_parser("evaluate", help="Run paid RAGAS judging on generated experiment answers")
    p.add_argument("--provider",choices=["nebius","groq","openai"],default="nebius")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--judge-model", help="Overrides the selected provider judge/model settings")
    p.add_argument("--output", type=Path, default=Path("runs/evaluation.json"))
    p = commands.add_parser("publish", help="Upsert dense vectors and verify Pinecone storage")
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--namespace")
    p.add_argument("--create", action="store_true", help="Create the index only if absent")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--output", type=Path, default=Path("runs/pinecone-publish.json"))
    return root


def main():
    """Context: errors are actionable, without leaking credentials or dumping tracebacks."""
    args = parser().parse_args()
    load_environment()
    try:
        if args.command == "demo":
            demo(args.output)
        elif args.command == "ingest":
            documents = load_documents(args.output) if args.output.exists() else []
            doc = parse_document(args.file, args.ticker, args.year, args.quarter, args.doc_type,
                                 args.source, args.accession, args.filing_date)
            documents = [old for old in documents if old.id != doc.id] + [doc]
            save_documents(documents, args.output)
            print(f"Saved {doc.id}: {len(doc.sections)} blocks; {len(documents)} documents total.")
        elif args.command == "download":
            download_sec(args.url, args.output, os.getenv("SEC_USER_AGENT", ""))
            print(f"Downloaded to {args.output}")
        elif args.command == "index":
            if args.strategy == "semantic" and args.backend != "dense":
                raise ValueError("Semantic chunking needs --backend dense.")
            embedder = get_embedder(args.embedding_model, args.revision) if args.backend == "dense" else None
            index = build_index(load_documents(args.documents), args.strategy, embedder)
            index.save(args.output)
            print(json.dumps(index.manifest, indent=2))
        elif args.command == "ask":
            index = Index.load(args.index)
            search_backend = None
            if args.retrieval == "pinecone":
                from .cloud import PineconeSearch
                if not index.vectors:
                    raise ValueError("Build a dense index before requesting Pinecone retrieval.")
                search_backend = PineconeSearch(args.pinecone_index or os.getenv("PINECONE_INDEX", "financial-rag"),
                                                args.namespace)
            embedder = get_embedder(index.manifest["embedding_model"], index.manifest["embedding_revision"]) if index.vectors else None
            filters = {key: value for key, value in {
                "ticker": args.ticker.upper() if args.ticker else None,
                "fiscal_year": args.year, "fiscal_quarter": args.quarter, "doc_type": args.doc_type,
            }.items() if value is not None}
            chain = build_rag_chain(index, embedder, filters, args.candidates,
                                    args.top_n, args.rerank, args.expand, args.provider, args.model,
                                    search_backend=search_backend)
            result = chain.invoke({"question": args.question})
            result.update(index_manifest=index.manifest, filters=filters,
                          pinecone_index=search_backend.name if search_backend else None,
                          namespace=search_backend.namespace if search_backend else None)
            write_json(args.output, result)
            print(result["answer"])
            print(f"\nAudit record: {args.output}")
        elif args.command == "experiment":
            from .evaluation import run_experiments
            fixed, semantic = Index.load(args.fixed), Index.load(args.semantic)
            embedder = get_embedder(fixed.manifest["embedding_model"], fixed.manifest["embedding_revision"])
            questions = json.loads(args.questions.read_text(encoding="utf-8"))
            rows = run_experiments(fixed, semantic, questions, embedder, args.output, args.provider, args.model,
                                   args.retrieval, args.pinecone_index,resume=args.resume)
            print(f"Saved {len(rows)} question/configuration results to {args.output}")
        elif args.command == "evaluate":
            from .evaluation import evaluate_ragas
            evaluate_ragas(args.run, args.output, args.judge_model, provider=args.provider)
            print(f"Saved RAGAS results to {args.output}")
        elif args.command == "publish":
            from .cloud import publish
            receipt = publish(Index.load(args.index), args.name, args.namespace,
                              create=args.create, region=args.region)
            write_json(args.output, receipt)
            print(json.dumps(receipt, indent=2))
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0

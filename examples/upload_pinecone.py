"""Context: build real fixed/semantic embeddings and upload to separate namespaces.

No data is deleted. Check-only performs a read-only account/configuration check.
Actual execution stores local snapshots and remote upload verification receipts.
"""

import argparse
import json
import os
from pathlib import Path

from financial_rag.cloud import inspect_service, publish
from financial_rag.core import Index, load_documents, stable_id
from financial_rag.generation import load_environment


def main():
    """Context: one corpus and one embedding model are shared across both builds."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--revision")
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()
    load_environment()
    name = args.name or os.getenv("PINECONE_INDEX", "financial-rag")
    Path("runs").mkdir(exist_ok=True)
    receipts = []
    try:
        service = inspect_service(name)
        print(json.dumps(service, indent=2), flush=True)
        if args.check_only:
            return 0
        if not service["exists"] and not args.create:
            raise ValueError("Target index absent; pass --create to create it.")
        if service["exists"] and (service["dimension"] != 768 or service["metric"] != "cosine"):
            raise ValueError("Target index is incompatible with BGE 768-dimensional cosine vectors.")
        from financial_rag.embedding import Embedder
        from financial_rag.chunking import build_index, CHUNKING_IMPLEMENTATION
        print("Loading one BGE model for both strategies...", flush=True)
        embedder = Embedder(revision=args.revision)
        documents = load_documents(Path("data/documents.json"))
        for strategy in ("fixed", "semantic"):
            path = Path(f"data/{strategy}.json")
            cached = Index.load(path) if path.exists() else None
            # Context: reuse only matching corpus/model snapshots when resuming.
            reusable = cached and cached.vectors and embedder.revision is not None \
                and cached.manifest["embedding_model"] == embedder.name \
                and cached.manifest["embedding_revision"] == embedder.revision \
                and cached.manifest["strategy"] == strategy \
                and cached.manifest.get("chunking_implementation") == CHUNKING_IMPLEMENTATION \
                and [item["hash"] for item in cached.manifest["corpus"]] == [doc.content_hash for doc in documents] \
                and [item.get("canonical_hash") for item in cached.manifest["corpus"]] == \
                    [stable_id(*(section.text for section in doc.sections)) for doc in documents]
            if reusable:
                index = cached
                print(f"Reusing {path}: {len(index.chunks)} vectors.", flush=True)
            else:
                print(f"Building {strategy} index...", flush=True)
                index = build_index(documents, strategy, embedder)
                index.save(path)
                print(f"Saved {path}: {len(index.chunks)} vectors.", flush=True)
            receipts.append(publish(index, name, namespace=strategy, create=args.create, region=args.region))
            Path("runs/pinecone-upload-report.json").write_text(json.dumps(receipts, indent=2), encoding="utf-8")
        Path("runs/pinecone-upload-status.json").write_text(json.dumps({
            "status": "uploaded", "index": name, "verification": "passed",
            "records_uploaded": sum(item["uploaded"] for item in receipts),
            "namespaces": [item["namespace"] for item in receipts]}, indent=2), encoding="utf-8")
        print(json.dumps(receipts, indent=2))
        return 0
    except Exception as exc:
        if not args.check_only:
            Path("runs/pinecone-upload-status.json").write_text(json.dumps({
                "status": "incomplete", "index": name, "completed_namespaces": receipts,
                "error_type": type(exc).__name__}, indent=2), encoding="utf-8")
        if isinstance(exc, (ValueError, RuntimeError)):
            print(f"Upload stopped: {exc}", flush=True)
        else:
            print(f"Upload stopped: {type(exc).__name__}; status {getattr(exc, 'status', None)}.", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

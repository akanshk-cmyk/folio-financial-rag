"""Context: local demo helpers reuse the production parsing and LCEL answer path.

Importing this module never downloads weights, accesses a database, or calls Groq.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import tempfile
from uuid import uuid4

from .chains import build_rag_chain
from .chunking import build_index
from .core import Index, load_documents
from .ingest import parse_document


ROOT = Path(__file__).resolve().parents[1]


def supporting_evidence(result):
    """Context: show only cited answer/operand excerpts; retain all passages in audit.

    Accept a quotation only when it occurs in the saved evidence. This display
    helper does not perform another search or invent a replacement excerpt.
    """
    references = [source for claim in result.get("claims", []) for source in claim.get("sources", [])]
    references.extend(operand for calc in result.get("calculations", []) for operand in calc.get("inputs", []))
    grouped = {}
    for reference in references:
        identifier, quote = reference.get("evidence_id"), reference.get("quote")
        passage = result.get("evidence", {}).get(identifier, "")
        if not isinstance(quote, str) or not quote.strip() or quote not in passage:
            continue
        entry = grouped.setdefault(identifier, {"identifier": identifier, "quotes": [],
                                  "source": result.get("sources", {}).get(identifier, {})})
        if quote not in entry["quotes"]:
            entry["quotes"].append(quote)
    return list(grouped.values())


def available_indexes():
    """Context: offer only valid index snapshots, never document or answer JSON files."""
    found = []
    candidates=list((ROOT / "data").glob("*.json"))
    candidates.extend((ROOT / "data/uploads").glob("*/index.json"))
    for path in sorted(candidates):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and all(key in data for key in ("manifest", "chunks", "parents")):
                found.append(path)
        except (ValueError, OSError):
            continue
    preferred = ROOT / "data/apple-2024-2025-lexical.json"
    return sorted(found, key=lambda path: (path != preferred, path.name))


def index_uploaded_pdf(content, ticker, fiscal_year, source_url="", current=None, *,
                       company_name="", currency="USD", embedder=None, strategy="enhanced",
                       storage_dir=None):
    """Context: preserve uploaded page provenance; merge only after parsing succeeds.

    Optional dense embeddings use the same encoder and LCEL retrieval as Q&A.
    A saved index is the commit marker: invalid PDFs never become discoverable.
    """
    if not content.startswith(b"%PDF-"):
        raise ValueError("Upload a PDF file.")
    if len(content) > 50 * 1024 * 1024:
        raise ValueError("PDF exceeds the 50 MB demo limit.")
    ticker = ticker.strip().upper()
    if not ticker or len(ticker) > 12 or not all(char.isalnum() or char in ".-" for char in ticker):
        raise ValueError("Enter a ticker of up to 12 letters, digits, dots, or hyphens.")
    if isinstance(fiscal_year,bool) or not isinstance(fiscal_year,int) or not 1990<=fiscal_year<=2100:
        raise ValueError("Enter a fiscal year between 1990 and 2100.")
    currency=currency.strip().upper()
    if currency not in {"USD","EUR","GBP","CAD","JPY","CNY","AUD","CHF","INR"}:
        raise ValueError("Select a supported reporting currency.")
    if strategy not in {"enhanced","fixed","semantic"}:
        raise ValueError("Select a supported chunk strategy.")
    if current is not None:
        expected=embedder.name if embedder else None
        revision=embedder.revision if embedder else None
        if (current.manifest.get("embedding_model")!=expected or
            current.manifest.get("embedding_revision")!=revision or
            current.manifest.get("strategy")!=strategy or bool(current.vectors)!=bool(embedder)):
            raise ValueError("Collection chunking and embedding settings must match the upload settings.")
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "uploaded.pdf"
        path.write_bytes(content)
        # A temporary filesystem path is not a durable source citation.
        source = source_url.strip() or "uploaded-pdf:" + hashlib.sha256(content).hexdigest()[:16]
        document = parse_document(path, ticker, fiscal_year, source=source)
    fresh = build_index([document], strategy, embedder)
    metadata={"company_name":company_name.strip()[:160] or ticker,"currency":currency,
              "currency_basis":"user-confirmed upload metadata","fiscal_year":fiscal_year,"ticker":ticker}
    upload_documents={**(current.manifest.get("upload_documents",{}) if current else {}),document.id:metadata}
    if current is not None:
        remaining = [chunk for chunk in current.chunks if chunk.doc_id != document.id]
        vectors=[vector for chunk,vector in zip(current.chunks,current.vectors) if chunk.doc_id!=document.id]+fresh.vectors
        parents = {chunk.parent_id: current.parents[chunk.parent_id] for chunk in remaining}
        parents.update(fresh.parents)
        fresh = Index({**fresh.manifest, "chunks": len(remaining) + len(fresh.chunks),
                       "documents": len({chunk.doc_id for chunk in remaining} | {document.id}),
                       "corpus": [item for item in current.manifest.get("corpus", []) if item["id"] != document.id]
                                 + fresh.manifest["corpus"]}, remaining + fresh.chunks, parents,vectors)
    fresh.manifest["upload_documents"]=upload_documents
    if storage_dir is not None:
        # Context: content/config-derived directories avoid trusting uploaded filenames or overwriting originals.
        identity=hashlib.sha256(json.dumps({"manifest":fresh.manifest,
            "chunks":[chunk.id for chunk in fresh.chunks]},sort_keys=True).encode()).hexdigest()[:20]
        folder=Path(storage_dir)/f"{ticker}-{fiscal_year}-{identity}"
        folder.mkdir(parents=True,exist_ok=True)
        artifact=folder/"source.pdf"
        artifact.write_bytes(content)
        document.provenance["local_artifact"]=str(artifact.resolve())
        (folder/"document.json").write_text(json.dumps([asdict(document)],indent=2),encoding="utf-8")
        fresh.manifest["saved_index"]=str((folder/"index.json").resolve())
        fresh.save(folder/"index.json")
    return fresh


def run_local_question(index, question, ticker, years, provider="extractive", rerank=False,
                       candidates=10, top_n=3, embedder=None, model=None, *, adaptive=False,
                       search_backend=None, namespace=None, on_event=None):
    """Context: UI selections become explicit metadata filters and captured settings."""
    filters = {"ticker": ticker}
    if years:
        filters["fiscal_year"] = years
    if adaptive:
        from .adaptive import adaptive_answer
        result=adaptive_answer(index,question,ticker=ticker,years=years,provider=provider,model=model,
            embedder=embedder,rerank=rerank,candidates=candidates,top_n=top_n,
            search_backend=search_backend,namespace=namespace,on_event=on_event)
    else:
        result = build_rag_chain(index, embedder, filters, candidates, top_n,
                                 rerank, True, provider, model, search_backend=search_backend).invoke({"question": question})
    result.update(filters=filters, index_manifest=index.manifest,
                  demo_settings={"candidates": candidates, "top_n": top_n, "rerank": rerank})
    return result


def save_answer(result):
    """Context: persist the exact answer and evidence without writing credentials."""
    folder = ROOT / "runs/streamlit"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"answer-{uuid4().hex}.json"
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return path


def postgres_catalog(store):
    """Context: inspect an existing notebook schema without creating or resetting it."""
    with store.connection() as connection:
        documents = connection.execute("SELECT * FROM financial_documents ORDER BY fiscal_year").fetchall()
        datasets = connection.execute("SELECT dataset_id,strategy,settings,count(*) AS chunks "
            "FROM financial_chunks GROUP BY dataset_id,strategy,settings ORDER BY dataset_id").fetchall()
    return documents, datasets

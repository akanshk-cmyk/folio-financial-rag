"""Context: single-document QA over persisted local dense-vector snapshots.

A source URL identifies an already indexed PDF; it is not fetched by loading.
Use index_pdf for a downloaded PDF that has no stored embeddings yet.
"""

import json
import math
import os
from pathlib import Path

from .chains import build_rag_chain
from .chunking import build_index
from .core import Index, stable_id
from .embedding import Embedder, MODEL
from .generation import load_environment, resolve_model
from .ingest import parse_document
from .retrieval import retrieve


class DocumentKnowledgeBase:
    """Context: expose search while keeping all retrieval inside the selected PDF."""

    def __init__(self, index, embedder, candidates, top_n, rerank):
        self.index, self.embedder = index, embedder
        self.candidates, self.top_n, self.rerank = candidates, top_n, rerank

    def search(self, question: str):
        """Context: return ranked passages with parent context and original source metadata."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Enter a nonempty question.")
        hits, _ = retrieve(self.index, question, self.embedder, candidates=self.candidates,
                           top_n=self.top_n, rerank=self.rerank, expand=True)
        return hits


class DocumentQA:
    """Context: initialize an open-source encoder and prepare source-isolated LCEL QA.

    vector_db_path is an existing JSON vector snapshot or a directory of snapshots,
    including upload subdirectories. This local adapter does not connect to Pinecone.
    An injected embedder enables offline testing or reuse of already loaded weights.
    """

    def __init__(self, vector_db_path, *, embedding_model=MODEL, revision=None,
                 provider="nebius", model=None, embedder=None, candidates=10, top_n=3,
                 rerank=False, max_context_chars=12000):
        if provider not in {"groq", "openai", "nebius", "extractive"}:
            raise ValueError("Select nebius, groq, openai, or extractive.")
        if not 1 <= top_n <= candidates or max_context_chars < 1:
            raise ValueError("Require candidates >= top_n >= 1 and a positive context budget.")
        load_environment()
        self.vector_db_path = Path(vector_db_path)
        self.embedder = embedder if embedder is not None else Embedder(embedding_model, revision)
        self.provider, self.model = provider, resolve_model(model, provider=provider)
        self.candidates, self.top_n, self.rerank = candidates, top_n, rerank
        self.max_context_chars = max_context_chars
        self.current_knowledge_base = None
        self.agent = None
        self.current_document = None

    def _activate(self, index, snapshot, source_url, document_id):
        """Context: validate model compatibility and vector alignment before replacing state."""
        if (index.manifest.get("embedding_model") != self.embedder.name or
                index.manifest.get("embedding_revision") != self.embedder.revision):
            raise ValueError("Embedding model/revision differs from the stored vectors. Load the recorded model revision.")
        if not index.vectors or len(index.vectors) != len(index.chunks):
            raise ValueError("Dense vectors are missing or misaligned. Index the PDF with index_pdf first.")
        selected = [i for i, chunk in enumerate(index.chunks)
                    if chunk.doc_id == document_id and chunk.source == source_url]
        chunks = [index.chunks[i] for i in selected]
        vectors = [index.vectors[i] for i in selected]
        widths = {len(vector) for vector in vectors}
        if not chunks or len(widths) != 1 or 0 in widths or any(
                not math.isfinite(value) for vector in vectors for value in vector):
            raise ValueError("Selected document has invalid dense vectors.")
        if any(chunk.parent_id not in index.parents for chunk in chunks):
            raise ValueError("Selected document is missing parent evidence.")
        scoped = Index({**index.manifest, "documents": 1, "chunks": len(chunks),
                        "corpus": [item for item in index.manifest.get("corpus", [])
                                   if item.get("id") == document_id]},
                       chunks, {chunk.parent_id: index.parents[chunk.parent_id] for chunk in chunks}, vectors)
        knowledge_base = DocumentKnowledgeBase(scoped, self.embedder, self.candidates, self.top_n, self.rerank)
        chain = build_rag_chain(scoped, self.embedder, candidates=self.candidates, top_n=self.top_n,
                                rerank=self.rerank, expand=True, provider=self.provider,
                                model=self.model, max_context_chars=self.max_context_chars)
        self.current_knowledge_base, self.agent = knowledge_base, chain
        self.current_document = {"source_url": source_url, "document_id": document_id,
                                 "snapshot": str(snapshot.resolve())}
        return dict(self.current_document)

    def load_document(self, pdf_url: str, *, document_id=None):
        """Context: resolve the exact stored source URL; never fall back to another PDF."""
        self.current_knowledge_base = self.agent = self.current_document = None
        if not isinstance(pdf_url, str) or not pdf_url.strip():
            raise ValueError("Provide the PDF's stored source URL.")
        source_url = pdf_url.strip()
        paths = [self.vector_db_path] if self.vector_db_path.is_file() else sorted(self.vector_db_path.rglob("*.json"))
        matches = []
        for path in paths:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict) or not {"manifest", "chunks", "parents"} <= data.keys():
                continue
            ids = {chunk["doc_id"] for chunk in data["chunks"] if chunk.get("source") == source_url
                   and (document_id is None or chunk.get("doc_id") == document_id)}
            for identity in ids:
                matches.append((path, identity))
        if not matches:
            raise FileNotFoundError("No stored vectors match this source URL. Index the downloaded PDF first.")
        if len(matches) != 1:
            raise ValueError("Multiple stored versions match. Select an exact snapshot path and, if needed, document_id.")
        snapshot, identity = matches[0]
        return self._activate(Index.load(snapshot), snapshot, source_url, identity)

    def index_pdf(self, pdf_path, *, source_url, ticker, fiscal_year, strategy="enhanced"):
        """Context: parse a downloaded PDF, embed it, persist vectors, then activate QA.

        Explicit metadata is required; a filename or URL never supplies a guessed year.
        Existing vector snapshots are not overwritten by this convenience method.
        """
        self.current_knowledge_base = self.agent = self.current_document = None
        if self.vector_db_path.suffix.lower() == ".json":
            raise ValueError("Use a directory vector_db_path when indexing a new PDF.")
        path = Path(pdf_path)
        if path.suffix.lower() != ".pdf" or not path.read_bytes().startswith(b"%PDF-"):
            raise ValueError("Provide a downloaded PDF file.")
        if not isinstance(source_url,str) or not source_url.strip() or not ticker.strip():
            raise ValueError("Provide the source URL and ticker.")
        if isinstance(fiscal_year,bool) or not isinstance(fiscal_year,int) or not 1990 <= fiscal_year <= 2100:
            raise ValueError("Provide an explicit fiscal year between 1990 and 2100.")
        document = parse_document(path, ticker, fiscal_year, source=source_url.strip())
        index = build_index([document], strategy, self.embedder)
        name = stable_id(document.id, source_url.strip(), self.embedder.name,
                         self.embedder.revision or "", strategy)
        snapshot = self.vector_db_path / f"{name}.json"
        if snapshot.exists():
            return self._activate(Index.load(snapshot), snapshot, source_url.strip(), document.id)
        index.save(snapshot)
        return self._activate(index, snapshot, source_url.strip(), document.id)

    def ask(self, question: str, *, adaptive=False):
        """Ask a question about the loaded document and return its answer/evidence audit."""
        if self.current_knowledge_base is None or self.agent is None:
            raise RuntimeError("Please load a document first!")
        if not isinstance(question,str) or not question.strip():
            raise ValueError("Enter a nonempty question.")
        load_environment()
        if self.provider != "extractive" and not os.getenv(self.provider.upper()+"_API_KEY"):
            raise ValueError("Set "+self.provider.upper()+"_API_KEY in the environment or .env.")
        if adaptive:
            from .adaptive import adaptive_answer
            scoped=self.current_knowledge_base.index
            result=adaptive_answer(scoped,question,ticker=scoped.chunks[0].ticker,
                years=sorted({chunk.fiscal_year for chunk in scoped.chunks}),provider=self.provider,
                model=self.model,embedder=self.embedder,rerank=self.rerank,
                candidates=self.candidates,top_n=self.top_n)
        else:
            result = self.agent.invoke({"question": question})
        from .demo_ui import supporting_evidence
        return {**result, "document": dict(self.current_document),
                "supporting_evidence": supporting_evidence(result)}


# Context: support the spelling in the original request; DocumentQA is preferred.
DocumemtQA = DocumentQA

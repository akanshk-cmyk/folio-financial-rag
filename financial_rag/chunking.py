"""Context: baseline chunking and structure-aware parent-child chunking are separate.

The fixed baseline intentionally measures characters, matching the attachment.
The semantic threshold is a percentile of distances, never a distance of 0.65.
"""

import platform
from importlib.metadata import version

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .core import Chunk, Index, stable_id
from .embedding import LangChainEmbeddings


CHUNKING_IMPLEMENTATION = "langchain-splitters-v1"


def fixed_chunks(text: str, size=512, overlap=64):
    """Context: LangChain recursively prefers paragraphs, lines, words, then characters.

    len() makes 512/64 character counts, not token counts. Overlap is a target
    dependent on separator boundaries, not a guaranteed duplicate suffix length.
    """
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("Require size > overlap >= 0.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap,
                                             length_function=len, is_separator_regex=False)
    return splitter.split_text(text)


def semantic_chunks(text, embedder, threshold=65):
    """Context: use the actual experimental SemanticChunker, not a local imitation.

    Breakpoints use the 65th percentile of adjacent sentence-window distances.
    A secondary recursive split caps oversized groups at 8,000 characters for storage.
    """
    if not 0 <= threshold <= 100:
        raise ValueError("Semantic percentile must be between 0 and 100.")
    if not text.strip():
        return []
    try:
        from langchain_experimental.text_splitter import SemanticChunker
    except ImportError as exc:
        raise RuntimeError("Install core LangChain dependencies with uv sync.") from exc
    splitter = SemanticChunker(embeddings=LangChainEmbeddings(embedder), buffer_size=1,
                               breakpoint_threshold_type="percentile",
                               breakpoint_threshold_amount=threshold)
    groups = splitter.split_text(text)
    return [part for group in groups for part in fixed_chunks(group, 8000, 0)]


def build_index(documents, strategy="fixed", embedder=None):
    """Context: a manifest records corpus identity and choices for later comparisons."""
    if strategy not in {"fixed", "semantic", "enhanced"}:
        raise ValueError("Unknown chunk strategy.")
    if strategy == "semantic" and embedder is None:
        raise ValueError("Semantic chunking requires model embeddings; lexical simulation is not semantic.")
    chunks, parents = [], {}
    for doc in documents:
        if strategy in {"fixed", "semantic"}:
            # Context: baseline chunks can span section boundaries, as in the walkthrough.
            text = "\n\n".join(section.text for section in doc.sections)
            pieces = list(fixed_chunks(text)) if strategy == "fixed" else semantic_chunks(text, embedder)
            for number, piece in enumerate(pieces):
                parents[stable_id(doc.id, strategy, str(number))] = piece
                parent_id = stable_id(doc.id, strategy, str(number))
                chunks.append(Chunk(stable_id(parent_id, piece), parent_id, doc.id, piece,
                                    doc.ticker, doc.fiscal_year, doc.fiscal_quarter, doc.doc_type,
                                    "Document", f"Chunk {number + 1}", doc.source, doc.synthetic))
        else:
            # Context: enhanced parents group adjacent blocks within a heading.
            groups, current = [], []
            length = 0
            for section in doc.sections:
                if current and (section.section_path != current[0].section_path or length > 5000
                                or section.location.startswith("PDF Page ")):
                    groups.append(current)
                    current, length = [], 0
                current.append(section)
                length += len(section.text)
            if current:
                groups.append(current)
            for group in groups:
                parent = "\n\n".join(s.text for s in group)
                parent_id = stable_id(doc.id, *(s.id for s in group))
                parents[parent_id] = parent
                for section in group:
                    # Context: tables remain whole; prose children target ~900 characters.
                    pieces = [section.text] if ("<table>" in section.text or
                                                section.location.startswith("PDF Page ")) else fixed_chunks(section.text, 900, 100)
                    for number, piece in enumerate(pieces):
                        chunks.append(Chunk(stable_id(parent_id, section.id, str(number), piece),
                                            parent_id, doc.id, piece, doc.ticker, doc.fiscal_year,
                                            doc.fiscal_quarter, doc.doc_type, section.section_path,
                                            section.location, doc.source, doc.synthetic))
    vectors = embedder.encode([chunk.text for chunk in chunks]) if embedder else []
    manifest = {
        "schema_version": 2, "strategy": strategy, "python": platform.python_version(),
        "chunking_implementation": CHUNKING_IMPLEMENTATION,
        "splitter": ("SemanticChunker" if strategy == "semantic" else
                     "RecursiveCharacterTextSplitter" if strategy == "fixed" else "structure-aware"),
        "langchain_text_splitters_version": version("langchain-text-splitters"),
        "langchain_experimental_version": version("langchain-experimental") if strategy == "semantic" else None,
        "corpus": [{"id": d.id, "hash": d.content_hash,
                    "canonical_hash": stable_id(*(s.text for s in d.sections)),
                    "parser": d.provenance.get("parser"),
                    "parser_version": d.provenance.get("parser_version")}
                   for d in documents],
        "embedding_model": embedder.name if embedder else None,
        "embedding_revision": embedder.revision if embedder else None,
        "embedding_window_policy": "mean-normalized-token-windows" if embedder else None,
        "query_prefix": "BGE retrieval instruction" if embedder else None,
        "fixed_size_chars": 512, "fixed_overlap_chars": 64,
        "semantic_percentile": 65, "semantic_buffer_sentences": 1,
        "backend": "dense" if embedder else "bm25-only-not-baseline-replication",
        "documents": len(documents), "chunks": len(chunks),
    }
    return Index(manifest, chunks, parents, vectors)

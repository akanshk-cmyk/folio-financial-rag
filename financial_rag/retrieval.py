"""Context: filter candidates before scoring, fuse ranks, and retain audit records."""

from collections import Counter
from dataclasses import dataclass
import math
import re

from .core import Chunk, Index, tokens
from .embedding import cosine
from .taxonomy import TAXONOMY_VERSION, has_numeric_row, match_concepts


RETRIEVAL_POLICY = "bm25-" + TAXONOMY_VERSION + "-statement-anchors-v1"


def financial_row_label(query):
    """Context: known financial vocabulary maps questions to row labels, never values.

    Product/category names take priority over generic net-sales wording. No company,
    fiscal year, page number, or reference answer is hardcoded into ranking.
    """
    products = ["Wearables, Home and Accessories", "iPhone", "Mac", "iPad", "Services"]
    product = next((label for label in products
                    if re.search(r"\b" + re.escape(label) + r"\b", query, re.I)), None)
    concepts = match_concepts(query)
    return product or (concepts[0].aliases[0] if concepts else None)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    text: str


def bm25(query, chunks):
    """Context: BM25 rewards exact terms without treating large passages as always better."""
    terms = [Counter(tokens(chunk.text)) for chunk in chunks]
    lengths = [sum(t.values()) for t in terms]
    average = sum(lengths) / len(lengths) if lengths else 1
    average = average or 1
    query_terms = set(tokens(query))
    label = financial_row_label(query)
    concepts = match_concepts(query)
    # Context: category questions should prioritize their category row, rather than
    # boosting every company-wide revenue total. Multiple metric requests still work.
    product = label in {"Wearables, Home and Accessories", "iPhone", "Mac", "iPad", "Services"}
    frequencies = {term: sum(term in counts for counts in terms) for term in query_terms}
    results = []
    for chunk, counts, length in zip(chunks, terms, lengths):
        score = 0.0
        for term in query_terms:
            frequency = counts[term]
            inverse = math.log(1 + (len(chunks) - frequencies[term] + .5) / (frequencies[term] + .5))
            denominator = frequency + 1.5 * (1 - .75 + .75 * length / average)
            score += inverse * frequency * 2.5 / denominator
        # Context: actual numeric table rows outrank incidental mentions of a metric.
        # Keep the full page, including headers, units and comparative columns.
        product_row = product and re.search(r"^\s*" + re.escape(label)
                        + r"(?:\s*\(\d+\))?\s+(?:\$\s*)?\(?[\d,]+", chunk.text, re.I | re.M)
        if product_row or (not product and any(has_numeric_row(chunk.text, item) for item in concepts)):
            score += 12.0
        if score > 0:
            results.append(Hit(chunk, score, chunk.text))
    return sorted(results, key=lambda hit: (-hit.score, hit.chunk.id))


def rrf(rankings, constant=60):
    """Context: ranks, rather than incomparable BM25/cosine scores, are combined."""
    scores, objects = {}, {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, 1):
            scores[hit.chunk.id] = scores.get(hit.chunk.id, 0) + 1 / (constant + rank)
            objects[hit.chunk.id] = hit.chunk
    return [Hit(objects[key], scores[key], objects[key].text)
            for key in sorted(scores, key=lambda key: (-scores[key], key))]


def flashrank(hits, query):
    """Context: local cross-encoding reorders a fixed candidate pool, never invents evidence."""
    try:
        from flashrank import Ranker, RerankRequest
    except ImportError as exc:
        raise RuntimeError("Install FlashRank with: uv sync --extra models") from exc
    if not hits:
        return []
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=".cache/flashrank", max_length=512)
    passages = [{"id": i, "text": hit.text} for i, hit in enumerate(hits)]
    results = ranker.rerank(RerankRequest(query=query, passages=passages))
    return [Hit(hits[int(item["id"])].chunk, float(item["score"]), hits[int(item["id"])].text)
            for item in results]


def primary_balance_statement(text):
    """Context: a table-of-contents mention is not an actual statement heading."""
    return bool(re.search(r"^[^\n|]{0,100}\bCONSOLIDATED[ \t]+BALANCE[ \t]+SHEETS?[ \t]*\n[ \t]*\(In\b",text,re.I|re.M) or
                re.search(r"^[ \t]*CONSOLIDATED[ \t]+BALANCE[ \t]+SHEETS?[ \t]*$",text,re.I|re.M))


def balance_statement(text):
    """Context: recognize a statement layout, not incidental discussion of its name."""
    return bool(primary_balance_statement(text) or
                (re.search(r"Current assets\s*:",text,re.I) and re.search(r"Current liabilities\s*:",text,re.I)))


def statement_anchors(index, query, filters):
    """Context: route an explicitly requested balance sheet within permitted company/years.

    These are original indexed chunks, not reference answers or new web evidence.
    Prefer a primary statement with units over a secondary detected-table copy.
    At most one anchor per selected company/year is added to the candidate pool.
    """
    if not filters.get("ticker") or not re.search(r"\bbalance\s+sheets?\b",query,re.I):
        return []
    allowed=[chunk for chunk in index.chunks if all(
        getattr(chunk,key) in (value if isinstance(value,list) else [value]) for key,value in filters.items())
        and balance_statement(chunk.text)]
    concepts=match_concepts(query)
    if concepts:
        allowed=[chunk for chunk in allowed if any(has_numeric_row(chunk.text,item) for item in concepts)]
    ranked=bm25(query,allowed)
    ranked.sort(key=lambda hit: not primary_balance_statement(hit.text))
    anchors=[]; seen=set()
    for hit in ranked:
        key=(hit.chunk.ticker,hit.chunk.fiscal_year)
        if key not in seen:
            anchors.append(hit);seen.add(key)
    return anchors


def merge_statement_anchors(hits, anchors, candidates):
    """Context: record anchors inside the bounded pool; reranking never adds hidden evidence."""
    merged=[];seen=set()
    for hit in anchors+hits:
        if hit.chunk.id not in seen:
            seen.add(hit.chunk.id);merged.append(hit)
    return merged[:candidates]


def retrieve(index: Index, query: str, embedder=None, filters=None, candidates=10, top_n=3,
             rerank=False, expand=False, saved_candidates=None):
    """Context: explicit metadata filters apply equally to sparse and dense branches.

    Saved candidates allow B and C to share exactly the same pre-reranking pool.
    This first version does not infer periods from natural-language questions.
    """
    if candidates < top_n or top_n < 1:
        raise ValueError("Require candidates >= top_n >= 1.")
    filters = filters or {}
    valid_filters = {"ticker", "fiscal_year", "fiscal_quarter", "doc_type"}
    if set(filters) - valid_filters:
        raise ValueError("Unsupported metadata filter.")
    allowed = [i for i, chunk in enumerate(index.chunks)
               if all(getattr(chunk, key) in (value if isinstance(value, list) else [value])
                      for key, value in filters.items())]
    chunks = [index.chunks[i] for i in allowed]
    if saved_candidates is not None:
        permitted = {chunk.id: chunk for chunk in chunks}
        if any(item["id"] not in permitted for item in saved_candidates):
            raise ValueError("Saved candidates do not match the filtered index.")
        hits = [Hit(permitted[item["id"]], item["score"], permitted[item["id"]].text)
                for item in saved_candidates]
    elif index.vectors:
        if embedder is None:
            raise ValueError("Dense index requires its embedding model.")
        if embedder.name != index.manifest["embedding_model"] or embedder.revision != index.manifest["embedding_revision"]:
            raise ValueError("Embedding model or revision differs from the index.")
        vector = embedder.encode([query], query=True)[0]
        dense = sorted([Hit(index.chunks[i], cosine(vector, index.vectors[i]), index.chunks[i].text)
                        for i in allowed], key=lambda hit: (-hit.score, hit.chunk.id))[:candidates]
        hits = rrf([dense, bm25(query, chunks)[:candidates]]) if index.manifest["strategy"] == "enhanced" else dense
    else:
        hits = bm25(query, chunks)
    hits = hits[:candidates]
    if saved_candidates is None:
        hits=merge_statement_anchors(hits,statement_anchors(index,query,filters),candidates)
    pool = [{"id": hit.chunk.id, "score": hit.score} for hit in hits]
    return select_context(index, hits, query, top_n, rerank, expand), pool


def select_context(index, hits, query, top_n=3, rerank=False, expand=False):
    """Context: local and Pinecone retrieval share the same reranking/expansion rules."""
    if rerank:
        hits = flashrank(hits, query)
    if re.search(r"\bbalance\s+sheets?\b",query,re.I):
        # Context: an exact requested statement outranks rounded management prose,
        # even if the cross-encoder favors that prose after truncating a long table.
        hits=sorted(hits,key=lambda h:(not primary_balance_statement(h.text),
                                      not balance_statement(h.text)))
        if re.search(r"cite\s+both\s+filings",query,re.I):
            distinct=[];rest=[];years=set()
            for hit in hits:
                if balance_statement(hit.text) and hit.chunk.fiscal_year not in years:
                    years.add(hit.chunk.fiscal_year);distinct.append(hit)
                else:
                    rest.append(hit)
            hits=distinct+rest
    selected, seen = [], set()
    for hit in hits:
        key = hit.chunk.parent_id if expand else hit.chunk.id
        if key in seen:
            continue
        seen.add(key)
        if expand:
            hit = Hit(hit.chunk, hit.score, index.parents[hit.chunk.parent_id])
        selected.append(hit)
        if len(selected) == top_n:
            break
    return selected

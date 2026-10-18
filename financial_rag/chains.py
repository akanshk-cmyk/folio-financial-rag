"""Context: assemble the actual retrieval-to-answer pipeline using LangChain LCEL.

Both CLI questions and controlled experiments call this same chain. No model
downloads or API requests occur while constructing it; execution starts at invoke().
"""

from langchain_core.runnables import RunnableLambda

from .generation import answer
from .retrieval import RETRIEVAL_POLICY, retrieve


def build_rag_chain(index, embedder=None, filters=None, candidates=10, top_n=3,
                    rerank=False, expand=False, provider="extractive", model=None,
                    saved_candidates=None, max_context_chars=24000, search_backend=None):
    """Context: keep query scope and retrieval settings explicit and reproducible."""

    def search(state):
        """Context: prefilter, retrieve, optionally rerank, then expand source context."""
        question = state.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("The RAG chain requires a nonempty question string.")
        search_function = search_backend or retrieve
        hits, pool = search_function(index, question, embedder, filters, candidates, top_n,
                                     rerank, expand, saved_candidates)
        return {"question": question, "hits": hits, "candidate_pool": pool}

    def synthesize(state):
        """Context: preview or grounded generation uses exactly the retrieved evidence."""
        result = answer(state["question"], state["hits"], provider, model, max_context_chars)
        return {**result, "candidate_pool": state["candidate_pool"], "orchestration": "langchain-lcel-v1",
                "retrieval_policy": RETRIEVAL_POLICY if search_backend is None else "external-backend-statement-anchors-v1",
                "retrieval_backend": "pinecone" if search_backend is not None else "local"}

    return (RunnableLambda(search).with_config(run_name="retrieve_rerank_expand")
            | RunnableLambda(synthesize).with_config(run_name="grounded_answer"))

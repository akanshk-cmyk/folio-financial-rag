"""Context: optional local models are imported only when dense retrieval is requested."""

import math
from pathlib import Path
import re
from langchain_core.embeddings import Embeddings


MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class LangChainEmbeddings(Embeddings):
    """Context: expose our existing Hugging Face encoder through LangChain's interface.

    SemanticChunker reuses the same model instance and document encoding policy
    as the vector index. Query instructions remain exclusive to query embeddings.
    """

    def __init__(self, embedder):
        self.embedder = embedder

    def embed_documents(self, texts):
        return self.embedder.encode(texts)

    def embed_query(self, text):
        return self.embedder.encode([text], query=True)[0]


def cosine(a, b):
    """Context: the same cosine measure is used for semantic boundaries and search."""
    numerator = sum(x * y for x, y in zip(a, b))
    denominator = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return numerator / denominator if denominator else 0.0


class Embedder:
    """Context: long texts use token windows, so their ends are not silently discarded.

    Window vectors are averaged and normalized. This documented choice differs
    from an original implementation that may have silently truncated long chunks.
    """

    def __init__(self, model=MODEL, revision=None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install model extras: uv sync --extra models") from exc
        # Context: resolve a mutable Hub branch once, then pin every experiment
        # and query to that commit. All downloaded weights stay in project cache.
        cache = ".cache/huggingface"
        if not Path(model).is_dir():
            from huggingface_hub import hf_hub_download
            configuration = hf_hub_download(model, "config.json", revision=revision, cache_dir=cache)
            commit = Path(configuration).parent.name
            if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
                raise RuntimeError("Could not resolve a pinned Hugging Face model revision.")
            revision = commit
        self.model = SentenceTransformer(model, revision=revision, cache_folder=cache, trust_remote_code=False)
        self.name = model
        self.revision = revision
        self.window_count = 0
        # Context: cap CPU inference threads to avoid oversubscription on laptops.
        if str(self.model.device).startswith("cpu"):
            import torch
            torch.set_num_threads(min(torch.get_num_threads(), 8))

    def encode(self, texts, query=False):
        tokenizer = self.model.tokenizer
        limit = self.model.max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)
        prefix = QUERY_PREFIX if query else ""
        prefix_size = len(tokenizer.encode(prefix, add_special_tokens=False))
        width = limit - prefix_size
        if width < 1:
            raise ValueError("Embedding input window has no available tokens.")
        windows, ranges = [], []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            start = len(windows)
            for offset in range(0, max(1, len(ids)), width):
                windows.append(prefix + tokenizer.decode(ids[offset:offset + width]))
            ranges.append((start, len(windows)))
        if not windows:
            return []
        vectors = self.model.encode(windows, normalize_embeddings=True, show_progress_bar=False, batch_size=16)
        self.window_count += len(windows)
        result = []
        for start, end in ranges:
            mean = [sum(float(vectors[i][j]) for i in range(start, end)) / (end - start)
                    for j in range(len(vectors[0]))]
            norm = math.sqrt(sum(x * x for x in mean)) or 1.0
            result.append([x / norm for x in mean])
        return result

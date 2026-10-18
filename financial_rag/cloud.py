"""Context: publish dense vectors with original text, then verify remote storage.

Local snapshots remain canonical. Stable IDs permit repeat uploads; namespaces
are never deleted and incompatible indexes are never replaced automatically.
"""

from dataclasses import asdict
import json
import math
import os
import time

from .core import stable_id
from .generation import load_environment


def client_from_environment():
    """Context: read local credentials without exposing them in logs or receipts."""
    load_environment()
    try:
        from pinecone import Pinecone
    except ImportError as exc:
        raise RuntimeError("Install cloud extras: uv sync --extra cloud") from exc
    if not os.getenv("PINECONE_API_KEY", "").strip():
        raise ValueError("Set PINECONE_API_KEY before publishing.")
    return Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def inspect_service(name):
    """Context: check account and target index without uploading or creating data."""
    client = client_from_environment()
    exists = name in client.list_indexes().names()
    result = {"index": name, "exists": exists}
    if exists:
        description = client.describe_index(name)
        result.update(dimension=description.dimension, metric=description.metric,
                      ready=bool(description.status["ready"]))
        stats=client.Index(host=description.host).describe_index_stats()
        result.update(total_vectors=stats.total_vector_count,
                      namespaces={key:value.vector_count for key,value in stats.namespaces.items()})
    return result


def prepare_records(index):
    """Context: preflight the whole snapshot before any remote writes occur."""
    if not index.chunks or len(index.vectors) != len(index.chunks):
        raise ValueError("Require one real dense vector for every chunk.")
    dimension = len(index.vectors[0])
    if dimension < 1 or len({chunk.id for chunk in index.chunks}) != len(index.chunks):
        raise ValueError("Vectors must be nonempty and chunk IDs unique.")
    dataset_id = stable_id(json.dumps(index.manifest, sort_keys=True))
    records = []
    for chunk, vector in zip(index.chunks, index.vectors):
        if len(vector) != dimension or any(not math.isfinite(value) for value in vector):
            raise ValueError("Vector dimensions must match and all values must be finite.")
        metadata = asdict(chunk)
        metadata.pop("id")
        metadata["dataset_id"] = dataset_id
        if len(json.dumps(metadata).encode("utf-8")) > 38_000:
            raise ValueError("Metadata too large; reduce chunk size before publishing.")
        records.append({"id": chunk.id, "values": vector, "metadata": metadata})
    return records, dimension, dataset_id


def publish(index, name, namespace=None, create=False, cloud="aws", region="us-east-1", verify=True):
    """Context: explicit creation, repeatable upserts, and representative read-back checks."""
    records, dimension, dataset_id = prepare_records(index)
    client = client_from_environment()
    if name not in client.list_indexes().names():
        if not create:
            raise ValueError("Pinecone index is absent; pass --create to create a compatible index.")
        from pinecone import ServerlessSpec
        client.create_index(name=name, dimension=dimension, metric="cosine",
                            spec=ServerlessSpec(cloud=cloud, region=region))
    deadline = time.monotonic() + 60
    description = client.describe_index(name)
    while not description.status["ready"]:
        if time.monotonic() > deadline:
            raise RuntimeError("Index is not ready yet; retry later. No data was deleted.")
        time.sleep(2)
        description = client.describe_index(name)
    if description.dimension != dimension or description.metric != "cosine":
        raise ValueError(f"Expected {dimension} dimensions and cosine. Existing index was not changed.")
    namespace = namespace or index.manifest["strategy"] + "-" + dataset_id[:12]
    target = client.Index(host=description.host)
    uploaded = 0
    for offset in range(0, len(records), 50):
        batch = records[offset:offset + 50]
        response = target.upsert(vectors=batch, namespace=namespace)
        if response.upserted_count != len(batch):
            raise RuntimeError("Incomplete batch acknowledgment; retry upload using stable IDs.")
        uploaded += response.upserted_count
        print(f"Pinecone {namespace}: {uploaded}/{len(records)} acknowledged.", flush=True)
    report = {"index": name, "namespace": namespace, "dataset_id": dataset_id,
              "dimension": dimension, "uploaded": uploaded, "verification": "not-requested"}
    if verify:
        # Context: eventual consistency can delay reads after an acknowledged upload.
        # Count checks and first/middle/last samples are not an exhaustive record audit.
        samples = [records[i] for i in sorted({0, len(records) // 2, len(records) - 1})]
        deadline = time.monotonic() + 60
        while True:
            fetched = target.fetch(ids=[record["id"] for record in samples], namespace=namespace)
            stats = target.describe_index_stats()
            summary = stats.namespaces.get(namespace)
            count = summary.vector_count if summary else 0
            if count >= len(records) and all(record["id"] in fetched.vectors for record in samples):
                break
            if time.monotonic() > deadline:
                raise RuntimeError("Upload acknowledged but read-back is pending; retry after consistency delay.")
            time.sleep(2)
        for record in samples:
            stored = fetched.vectors[record["id"]]
            if stored.metadata.get("text") != record["metadata"]["text"] or len(stored.values) != dimension:
                raise RuntimeError("Stored metadata/vector validation failed.")
        probe = target.query(vector=records[0]["values"], namespace=namespace, top_k=3,
                             filter={"dataset_id": {"$eq": dataset_id}}, include_metadata=True)
        if not probe.matches or any(not match.metadata.get("text") for match in probe.matches):
            raise RuntimeError("Read-back passed but vector query returned no text metadata.")
        report.update(verification="passed", namespace_vector_count=count,
                      sampled_records_verified=len(samples), vector_query_matches=len(probe.matches))
    return report


class PineconeSearch:
    """Context: retrieve actual remote vectors while validating against the local manifest.

    Text must match the local snapshot, so stale or foreign namespace records can
    never silently enter the answer. Dataset IDs and user filters are applied by
    Pinecone before retrieval. B/C can replay one saved candidate pool.
    """

    def __init__(self, name, namespace):
        if not name or not namespace:
            raise ValueError("Pinecone retrieval requires an index name and namespace.")
        self.name, self.namespace = name, namespace
        self.target = None

    def __call__(self, index, query, embedder=None, filters=None, candidates=10, top_n=3,
                 rerank=False, expand=False, saved_candidates=None):
        from .retrieval import (Hit, retrieve, select_context, bm25, rrf,
                                statement_anchors, merge_statement_anchors)
        if not index.vectors or embedder is None:
            raise ValueError("Pinecone retrieval needs a dense local snapshot and its Hugging Face encoder.")
        if candidates < top_n or top_n < 1:
            raise ValueError("Require candidates >= top_n >= 1.")
        if (embedder.name != index.manifest["embedding_model"] or
                embedder.revision != index.manifest["embedding_revision"]):
            raise ValueError("Pinecone query encoder differs from the snapshot's model/revision.")
        filters = filters or {}
        conditions = [{"dataset_id": {"$eq": stable_id(json.dumps(index.manifest, sort_keys=True))}}]
        for key, value in filters.items():
            if key not in {"ticker", "fiscal_year", "fiscal_quarter", "doc_type"}:
                raise ValueError("Unsupported metadata filter.")
            conditions.append({key: {"$in": value} if isinstance(value, list) else {"$eq": value}})
        # Context: C reuses B's retrieved candidates without making another search.
        if saved_candidates is not None:
            return retrieve(index, query, filters=filters, candidates=candidates, top_n=top_n,
                            rerank=rerank, expand=expand, saved_candidates=saved_candidates)
        known = {chunk.id: chunk for chunk in index.chunks}
        try:
            if self.target is None:
                client = client_from_environment()
                description = client.describe_index(self.name)
                if description.dimension != len(index.vectors[0]) or description.metric != "cosine":
                    raise ValueError("Pinecone index is incompatible with the local dense snapshot.")
                self.target = client.Index(host=description.host)
            response = self.target.query(vector=embedder.encode([query], query=True)[0],
                                         namespace=self.namespace, top_k=candidates,
                                         filter={"$and": conditions}, include_metadata=True)
        except (ValueError, RuntimeError):
            raise
        except Exception as exc:
            raise RuntimeError(f"Pinecone query failed: {type(exc).__name__}, status {getattr(exc, 'status', None)}.") from exc
        hits = []
        for match in response.matches:
            if match.id not in known or not match.metadata:
                raise ValueError("Remote candidate is absent from the selected local snapshot.")
            chunk = known[match.id]
            if match.metadata.get("text") != chunk.text:
                raise ValueError("Remote text differs from the local snapshot; republish before querying.")
            if any(getattr(chunk, key) not in (value if isinstance(value, list) else [value])
                   for key, value in filters.items()):
                raise ValueError("Pinecone returned a candidate outside the requested scope.")
            hits.append(Hit(chunk, float(match.score), match.metadata["text"]))
        if index.manifest["strategy"] == "enhanced":
            allowed = [chunk for chunk in index.chunks if all(
                getattr(chunk, key) in (value if isinstance(value, list) else [value])
                for key, value in filters.items())]
            hits = rrf([hits, bm25(query, allowed)[:candidates]])
        # Context: fetch explicitly requested statement anchors from this SAME
        # Pinecone dataset, then validate their stored text and scope. A local
        # snapshot match alone is never represented as remotely retrieved evidence.
        anchors=statement_anchors(index,query,filters)
        if anchors:
            try:
                fetched=self.target.fetch(ids=[hit.chunk.id for hit in anchors],namespace=self.namespace)
            except Exception as exc:
                raise RuntimeError("Pinecone statement fetch failed: "+type(exc).__name__) from exc
            for hit in anchors:
                record=fetched.vectors.get(hit.chunk.id)
                if record is None or not record.metadata or record.metadata.get("text")!=hit.chunk.text:
                    raise ValueError("Pinecone statement anchor differs from the selected local dataset.")
                if record.metadata.get("dataset_id")!=stable_id(json.dumps(index.manifest,sort_keys=True)):
                    raise ValueError("Pinecone statement anchor belongs to a different dataset.")
        hits = merge_statement_anchors(hits,anchors,candidates)
        pool = [{"id": hit.chunk.id, "score": hit.score} for hit in hits]
        return select_context(index, hits, query, top_n, rerank, expand), pool

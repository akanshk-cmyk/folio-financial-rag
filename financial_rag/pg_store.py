"""Context: small-corpus exact pgvector search plus PostgreSQL full-text search.

Full-text ranking uses ts_rank_cd, not BM25. MATERIALIZED eligible rows enforce
metadata scope before distance ranking. No ANN index is needed for two filings.
"""

from contextlib import contextmanager
import json
import math

from .core import Chunk, stable_id
from .retrieval import Hit, rrf, flashrank


SCHEMA = """
CREATE TABLE IF NOT EXISTS financial_documents (
 document_id text PRIMARY KEY, company text NOT NULL, fiscal_year integer NOT NULL,
 title text NOT NULL, source_url text NOT NULL, file_sha256 text NOT NULL
);
CREATE TABLE IF NOT EXISTS financial_chunks (
 chunk_id text PRIMARY KEY, document_id text NOT NULL REFERENCES financial_documents(document_id),
 strategy text NOT NULL CHECK (strategy IN ('fixed','semantic')),
 pdf_page integer NOT NULL CHECK (pdf_page > 0), dataset_id text NOT NULL,
 text text NOT NULL, embedding vector(768) NOT NULL,
 search_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
 settings jsonb NOT NULL
);
CREATE INDEX IF NOT EXISTS financial_chunks_fts ON financial_chunks USING gin(search_tsv);
CREATE INDEX IF NOT EXISTS financial_chunks_scope ON financial_chunks(strategy,dataset_id,document_id);
CREATE TABLE IF NOT EXISTS financial_runs (
 run_id text PRIMARY KEY, created_at timestamptz NOT NULL DEFAULT now(), payload jsonb NOT NULL
);
"""


class PGStore:
    """Context: connections are short-lived; repeated initialization preserves data."""

    def __init__(self, dsn):
        self.dsn = dsn

    @contextmanager
    def connection(self):
        import psycopg
        from psycopg.rows import dict_row
        from pgvector.psycopg import register_vector
        with psycopg.connect(self.dsn, row_factory=dict_row) as connection:
            register_vector(connection)
            yield connection

    def initialize(self):
        """Context: extension installation is an explicit administrator setup step."""
        with self.connection() as connection:
            connection.execute(SCHEMA)
            dimension = connection.execute("SELECT format_type(atttypid,atttypmod) AS type "
                "FROM pg_attribute WHERE attrelid='financial_chunks'::regclass AND attname='embedding'").fetchone()
            if dimension["type"] != "vector(768)":
                raise ValueError("Existing chunk table must use vector(768); do not silently reuse incompatible data.")

    def reset(self, confirmation):
        """Context: this explicit reset clears demo records only, never runs automatically."""
        if confirmation != "DELETE_DEMO_DATA":
            raise ValueError("Reset requires confirmation='DELETE_DEMO_DATA'.")
        with self.connection() as connection:
            connection.execute("TRUNCATE financial_runs, financial_chunks, financial_documents")

    def put_document(self, metadata):
        """Context: source identity upserts are idempotent and do not delete other filings."""
        values = metadata.model_dump()
        with self.connection() as connection:
            connection.execute("""INSERT INTO financial_documents
                (document_id,company,fiscal_year,title,source_url,file_sha256)
                VALUES (%(document_id)s,%(company)s,%(fiscal_year)s,%(title)s,%(source_url)s,%(file_sha256)s)
                ON CONFLICT(document_id) DO UPDATE SET title=EXCLUDED.title,source_url=EXCLUDED.source_url""", values)

    def put_chunks(self, chunks, vectors, settings):
        """Context: preflight every vector before a committed batch enters the database."""
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk requires one embedding.")
        if any(len(vector) != 768 or any(not math.isfinite(float(v)) for v in vector) for vector in vectors):
            raise ValueError("Embeddings must contain 768 finite values.")
        import numpy as np
        from psycopg.types.json import Jsonb
        rows = [(metadata.chunk_id, metadata.document_id, metadata.strategy, metadata.pdf_page,
                 metadata.dataset_id, text, np.asarray(vector, dtype=np.float32), Jsonb(settings))
                for (metadata, text), vector in zip(chunks, vectors)]
        with self.connection() as connection, connection.cursor() as cursor:
            cursor.executemany("""INSERT INTO financial_chunks
                (chunk_id,document_id,strategy,pdf_page,dataset_id,text,embedding,settings)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT(chunk_id) DO UPDATE SET text=EXCLUDED.text,embedding=EXCLUDED.embedding,
                settings=EXCLUDED.settings""", rows)

    def search(self, request, vector, dataset_id):
        """Context: both branches share strategy/company/year/dataset restrictions."""
        import numpy as np
        scope = "c.strategy=%s AND d.company=%s AND c.dataset_id=%s"
        parameters = [request.strategy, request.company, dataset_id]
        if request.fiscal_years is not None:
            scope += " AND d.fiscal_year=ANY(%s)"
            parameters.append(request.fiscal_years)
        eligible = ("SELECT c.chunk_id,c.document_id,c.pdf_page,c.text,c.search_tsv,"
                    "c.embedding,d.company,d.fiscal_year,d.title,d.source_url "
                    "FROM financial_chunks c JOIN financial_documents d USING(document_id) WHERE " + scope)
        dense_sql = ("WITH eligible AS MATERIALIZED (" + eligible + ") "
                     "SELECT *, 1-(embedding <=> %s) AS score FROM eligible "
                     "ORDER BY embedding <=> %s,chunk_id LIMIT %s")
        sparse_sql = ("WITH eligible AS MATERIALIZED (" + eligible + ") "
                      "SELECT *,ts_rank_cd(search_tsv,websearch_to_tsquery('english',%s)) AS score "
                      "FROM eligible WHERE search_tsv @@ websearch_to_tsquery('english',%s) "
                      "ORDER BY score DESC,chunk_id LIMIT %s")
        query_vector = np.asarray(vector, dtype=np.float32)
        with self.connection() as connection:
            dense = connection.execute(dense_sql, parameters + [query_vector, query_vector, request.candidates]).fetchall()
            sparse = connection.execute(sparse_sql, parameters + [request.question, request.question, request.candidates]).fetchall()
        return dense, sparse

    def save_run(self, payload):
        """Context: retain the exact evidence, settings, outputs, and audit for judging."""
        from psycopg.types.json import Jsonb
        from uuid import uuid4
        run_id = str(uuid4())
        with self.connection() as connection:
            connection.execute("INSERT INTO financial_runs(run_id,payload) VALUES (%s,%s)", (run_id, Jsonb(payload)))
        return run_id

    def counts(self):
        with self.connection() as connection:
            return connection.execute("SELECT strategy,count(*) AS chunks FROM financial_chunks GROUP BY strategy").fetchall()


def rows_to_hits(rows):
    """Context: database records retain PDF page provenance in the shared evidence type."""
    return [Hit(Chunk(row["chunk_id"], row["chunk_id"], row["document_id"], row["text"],
                      row["company"], row["fiscal_year"], "FY", "10-K", row["title"],
                      f"PDF Page {row['pdf_page']}", row["source_url"]), float(row["score"]), row["text"])
            for row in rows]


def fuse_candidates(dense_rows, sparse_rows, maximum):
    """Context: RRF combines ranking positions and deduplicates identical chunk IDs."""
    dense, sparse = rows_to_hits(dense_rows), rows_to_hits(sparse_rows)
    fused = rrf([dense, sparse])[:maximum]
    audit = [{"chunk_id": hit.chunk.id, "rrf_score": hit.score,
              "dense_rank": next((i for i,h in enumerate(dense,1) if h.chunk.id == hit.chunk.id), None),
              "keyword_rank": next((i for i,h in enumerate(sparse,1) if h.chunk.id == hit.chunk.id), None)}
             for hit in fused]
    return fused, audit

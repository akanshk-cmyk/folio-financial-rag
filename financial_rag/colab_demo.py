"""Context: notebook application for token chunking, pgvector hybrid RAG, and evaluation.

The existing Pinecone CLI is independent. This application uses local Postgres,
PyPDF, Hugging Face, FlashRank, Groq, and the shared financial validation code.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time

from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .core import stable_id
from .demo_schemas import DocumentMetadata, ChunkMetadata, QuestionInput, Citation, Calculation, AnswerResult
from .embedding import LangChainEmbeddings
from .generation import answer, resolve_model
from .pg_store import fuse_candidates
from .retrieval import flashrank


def cover_filing_years(hits, years, maximum):
    """Context: multi-filing questions reserve evidence slots for each requested year."""
    selected = []
    for year in years or []:
        hit = next((hit for hit in hits if hit.chunk.fiscal_year == year), None)
        if hit and hit.chunk.id not in {item.chunk.id for item in selected}:
            selected.append(hit)
    for hit in hits:
        if len(selected) >= maximum:
            break
        if hit.chunk.id not in {item.chunk.id for item in selected}:
            selected.append(hit)
    return selected


@dataclass
class ParsedPDF:
    metadata: DocumentMetadata
    pages: list[tuple[int, str]]
    quality_checks: list[dict]


def parse_uploaded_pdf(path, year, source_url):
    """Context: preserve physical PDF page numbers and enforce known statement checks.

    The reference values were checked against the supplied Apple PDFs, independent
    of extraction. New versions that disagree require review, not silent indexing.
    """
    from pypdf import PdfReader
    path = Path(path)
    reader = PdfReader(path)
    if reader.is_encrypted:
        raise ValueError("Provide an unlocked PDF.")
    pages, checks = [], []
    expected = {2024: {"Total net sales": ["391,035", "383,285", "394,328"],
                       "Operating income": ["123,216", "114,301", "119,437"],
                       "Net income": ["93,736", "96,995", "99,803"]},
                2025: {"Total net sales": ["416,161", "391,035", "383,285"],
                       "Operating income": ["133,050", "123,216", "114,301"],
                       "Net income": ["112,010", "93,736", "96,995"]}}
    if year not in expected:
        raise ValueError("This demonstration's quality gate supports the two verified Apple filing years only.")
    for number, page in enumerate(reader.pages, 1):
        text = page.extract_text(extraction_mode="layout") or ""
        lines = [line.rstrip() for line in text.splitlines()
                 if not re.match(r"^\s*\d{1,2}/\d{1,2}/\d{2,4},\s+\d{1,2}:\d{2}\s+[AP]M", line)
                 and not re.match(r"^\s*https?://www\.sec\.gov/Archives/.*\s+\d+/\d+\s*$", line)]
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
        if not text:
            if page.images:
                raise ValueError(f"Page {number} needs OCR; repair extraction before indexing.")
            continue
        pages.append((number, text))
        if "CONSOLIDATED STATEMENTS OF OPERATIONS" in text:
            for label, values in expected[year].items():
                row = next((line for line in text.splitlines() if re.match(r"^\s*"+re.escape(label)+r"\s+", line)), "")
                found = re.findall(r"\d[\d,]*", row)
                checks.append({"metric": label, "page": number, "expected": values,
                               "actual": found, "passed": found == values})
            checks.append({"metric": "statement units", "page": number,
                           "passed": "In millions" in text})
    if len(checks) != 4 or not all(check["passed"] for check in checks):
        raise ValueError("PDF table quality gate failed. Inspect/repair the statement extraction before indexing.")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = DocumentMetadata(document_id=f"AAPL_{year}_10K_{digest[:12]}", fiscal_year=year,
                                 title=f"Apple FY{year} Form 10-K", source_url=source_url, file_sha256=digest)
    return ParsedPDF(metadata, pages, checks)


def build_page_chunks(document, embedder, strategy, settings):
    """Context: this notebook's fixed baseline uses TOKENS, unlike the legacy CLI.

    Both strategies operate on the same page text, preserving exact page provenance.
    BGE's tokenizer supplies the length function. Long semantic groups are retained;
    the shared embedder windows them rather than silently truncating their endings.
    """
    if strategy not in {"fixed", "semantic"}:
        raise ValueError("Unknown chunking strategy.")
    token_length = lambda text: len(embedder.model.tokenizer.encode(text, add_special_tokens=True))
    fixed = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64, length_function=token_length)
    if strategy == "semantic":
        from langchain_experimental.text_splitter import SemanticChunker
        semantic = SemanticChunker(LangChainEmbeddings(embedder), buffer_size=1,
                                   breakpoint_threshold_type="percentile", breakpoint_threshold_amount=65)
    dataset_id = stable_id(json.dumps(settings, sort_keys=True), strategy)
    chunks = []
    for page, text in document.pages:
        pieces = fixed.split_text(text) if strategy == "fixed" else semantic.split_text(text)
        # Context: the secondary storage cap is independent of the token baseline.
        cap = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=0, length_function=len)
        pieces = [part for piece in pieces for part in (cap.split_text(piece) if len(piece) > 8000 else [piece])]
        for position, piece in enumerate(pieces):
            if not piece.strip():
                continue
            metadata = ChunkMetadata(chunk_id=stable_id(document.metadata.document_id,str(page),strategy,str(position),piece,dataset_id),
                                     document_id=document.metadata.document_id, pdf_page=page,
                                     strategy=strategy, dataset_id=dataset_id)
            chunks.append((metadata, piece))
    lengths = [token_length(text) for _,text in chunks]
    return chunks, {"strategy": strategy, "chunks": len(chunks), "maximum_tokens": max(lengths, default=0),
                    "over_embedding_limit": sum(length > embedder.model.max_seq_length for length in lengths),
                    "long_input_policy": "mean-normalized-token-windows", "dataset_id": dataset_id}


class FinancialDemo:
    """Context: one explicit evidence path serves generation, auditing, and judging."""

    def __init__(self, store, embedder, generation_model=None, generation_provider="groq"):
        self.store, self.embedder = store, embedder
        self.provider = generation_provider
        self.model = resolve_model(generation_model,provider=generation_provider)
        self.datasets = {}
        self.documents = {}

    def ingest(self, documents):
        """Context: all configurations share a pinned encoder and canonical corpus."""
        settings = {"embedding_model": self.embedder.name, "revision": self.embedder.revision,
                    "corpus": [doc.metadata.file_sha256 for doc in documents],
                    "fixed_tokens": 512, "overlap_tokens": 64, "semantic_percentile": 65,
                    "parser": "pypdf-layout", "implementation": "colab-pg-v1"}
        reports = []
        for strategy in ("fixed", "semantic"):
            dataset = stable_id(json.dumps(settings, sort_keys=True), strategy)
            for document in documents:
                self.store.put_document(document.metadata)
                self.documents[document.metadata.document_id] = document.metadata
                chunks, report = build_page_chunks(document, self.embedder, strategy, settings)
                vectors = self.embedder.encode([text for _,text in chunks])
                self.store.put_chunks(chunks, vectors, settings)
                reports.append({"document_id": document.metadata.document_id, **report})
            self.datasets[strategy] = dataset
        return reports

    def retrieve(self, request):
        """Context: exact dense and full-text ranks are fused before optional reranking."""
        request = QuestionInput.model_validate(request)
        if request.strategy not in self.datasets:
            raise ValueError("Ingest both documents before querying.")
        started = time.perf_counter()
        vector = self.embedder.encode([request.question], query=True)[0]
        dense, sparse = [], []
        for year in request.fiscal_years or [None]:
            scoped = request.model_copy(update={"fiscal_years": [year] if year is not None else None})
            dense_rows, sparse_rows = self.store.search(scoped, vector, self.datasets[request.strategy])
            dense.extend(dense_rows)
            sparse.extend(sparse_rows)
        dense.sort(key=lambda row: (-row["score"], row["chunk_id"]))
        sparse.sort(key=lambda row: (-row["score"], row["chunk_id"]))
        hits, audit = fuse_candidates(dense, sparse, max(request.candidates,len(dense)+len(sparse)))
        hits = cover_filing_years(hits, request.fiscal_years, request.candidates)
        ids = {hit.chunk.id for hit in hits}
        return {"hits": hits, "candidate_audit": [item for item in audit if item["chunk_id"] in ids],
                "retrieval_seconds": time.perf_counter()-started}

    def generate(self, request, retrieved):
        """Context: retained passages are precisely the ones provided to generation."""
        request = QuestionInput.model_validate(request)
        started = time.perf_counter()
        hits = list(retrieved["hits"])
        if request.rerank:
            hits = flashrank(hits, request.question)
        hits = cover_filing_years(hits, request.fiscal_years, request.top_n)
        raw = answer(request.question, hits, self.provider, self.model, request.context_chars)
        result = self.package(request.question, hits, raw, request.model_dump(),
                              retrieved["retrieval_seconds"] + time.perf_counter()-started)
        result.audit.update(candidate_audit=retrieved["candidate_audit"], retrieval_seconds=retrieved["retrieval_seconds"])
        missing_years = set(request.fiscal_years or []) - {hit.chunk.fiscal_year for hit in hits}
        result.audit["missing_requested_filing_years"] = sorted(missing_years)
        if missing_years:
            result.insufficient_evidence = True
        self.store.save_run(result.model_dump(mode="json"))
        return result

    def qa(self, request):
        """Context: LCEL makes retrieval and generation explicit, with no hidden second search."""
        request = QuestionInput.model_validate(request)
        chain = (RunnableLambda(lambda _: self.retrieve(request))
                 | RunnableLambda(lambda evidence: self.generate(request, evidence)))
        return chain.invoke({"question": request.question})

    def package(self, question, hits, raw, settings, latency):
        """Context: convert shared validated generation into the notebook's Pydantic result."""
        by_id = {hit.chunk.id: hit for hit in hits}
        citations = []
        used = {source["evidence_id"] for claim in raw.get("claims", []) for source in claim["sources"]}
        used |= {item["evidence_id"] for calc in raw["calculations"] for item in calc["inputs"]}
        context_metadata = {}
        for source_id, source in raw["sources"].items():
            hit = by_id[source["chunk_id"]]
            page = re.search(r"PDF Page (\d+)", hit.chunk.location)
            title = self.documents.get(hit.chunk.doc_id)
            citation = Citation(source_id=source_id, document_id=hit.chunk.doc_id,
                document_title=title.title if title else hit.chunk.section_path,
                pdf_page=int(page[1]) if page else None, web_url=hit.chunk.source)
            context_metadata[source_id] = {**citation.model_dump(), "filing_year": hit.chunk.fiscal_year}
            if source_id in used:
                citations.append(citation)
        calculations = [Calculation(formula=calc["formula"], source_values=calc["inputs"], units=calc["unit"],
                        result=calc["raw_output"], supporting_source_ids=list(dict.fromkeys(
                            item["evidence_id"] for item in calc["inputs"]))) for calc in raw["calculations"]]
        return AnswerResult(question=question, answer=raw["answer"], citations=citations, calculations=calculations,
                            insufficient_evidence=raw["status"] != "generated" or bool(raw.get("missing")) or raw["answer"] == "Information not available in provided financial disclosures.",
                            status=raw["status"], retrieved_passages=raw["evidence"], run_settings=settings,
                            latency_seconds=latency, audit={"generation": raw, "context_metadata": context_metadata})

    def compare(self, questions, output):
        """Context: four controlled configurations share corpus/model/prompt and hybrid search.

        Each strategy's fused candidate pool is reused for its reranked counterpart.
        Latency records explicitly mark reuse rather than implying a fresh search.
        """
        from .demo_evaluation import deterministic_checks
        rows = []
        for question in questions:
            for strategy, labels in (("fixed", ("A", "B")), ("semantic", ("C", "D"))):
                request = QuestionInput(question=question["question"], strategy=strategy, rerank=False,
                    fiscal_years=question.get("fiscal_years"), candidates=10, top_n=5, context_chars=24000)
                retrieved = self.retrieve(request)
                for label, rerank in zip(labels, (False, True)):
                    result = self.generate(request.model_copy(update={"rerank": rerank}), retrieved)
                    raw = result.audit["generation"]
                    rows.append({"configuration": label, "question_id": question["id"],
                        "reference": question["reference"], **raw,
                        "question": question["question"], "settings": result.run_settings,
                        "retrieval_reused": rerank, "seconds": result.latency_seconds,
                        "candidate_audit": retrieved["candidate_audit"],
                        "deterministic": deterministic_checks(result, question["reference"],
                            question.get("expected_evidence")),
                        "expected_evidence": question.get("expected_evidence", [])})
        manifest = {"backend": "postgresql-pgvector+full-text", "hybrid": True, "rrf_constant": 60,
                    "generator": self.model, "embedding_model": self.embedder.name,
                    "embedding_revision": self.embedder.revision, "datasets": self.datasets,
                    "comparison": {"A": "fixed", "B": "fixed+rerank", "C": "semantic", "D": "semantic+rerank"}}
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps({"manifest": manifest, "rows": rows}, indent=2), encoding="utf-8")
        return rows

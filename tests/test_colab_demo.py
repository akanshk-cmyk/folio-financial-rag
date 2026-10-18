"""Context: validate Colab code without installing a database or running live tools."""

import ast
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError

from financial_rag.colab_demo import FinancialDemo, ParsedPDF, build_page_chunks, parse_uploaded_pdf, cover_filing_years
from financial_rag.core import Chunk
from financial_rag.demo_evaluation import deterministic_checks
from financial_rag.demo_schemas import AnswerResult, Citation, DocumentMetadata, QuestionInput
from financial_rag.pg_store import PGStore, fuse_candidates
from financial_rag.research_agents import public_url
from financial_rag.retrieval import Hit


class ColabContractsTests(unittest.TestCase):
    """Context: source integrity and experiment settings are enforced before model calls."""

    def document(self):
        return DocumentMetadata(document_id="doc", fiscal_year=2024, title="Apple FY2024 Form 10-K",
                                source_url="https://www.sec.gov/Archives/example", file_sha256="hash")

    def encoder(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=True):
                return list(range(len(text.split()) + (2 if add_special_tokens else 0)))
        return SimpleNamespace(model=SimpleNamespace(tokenizer=Tokenizer(), max_seq_length=512))

    def test_fixed_baseline_measures_tokens_not_characters(self):
        document = ParsedPDF(self.document(), [(1, "financialword " * 900)], [])
        chunks, report = build_page_chunks(document, self.encoder(), "fixed", {"revision": "pinned"})
        self.assertTrue(chunks)
        self.assertTrue(any(len(text) > 512 for _,text in chunks))
        self.assertLessEqual(report["maximum_tokens"], 512)

    def test_page_ids_are_stable_and_never_span_pages(self):
        document = ParsedPDF(self.document(), [(1, "Same text"), (2, "Same text")], [])
        first, _ = build_page_chunks(document, self.encoder(), "fixed", {})
        second, _ = build_page_chunks(document, self.encoder(), "fixed", {})
        self.assertEqual([item.chunk_id for item,_ in first], [item.chunk_id for item,_ in second])
        self.assertEqual([item.pdf_page for item,_ in first], [1,2])
        self.assertNotEqual(first[0][0].chunk_id, first[1][0].chunk_id)

    def test_bad_question_limits_and_duplicate_years_are_rejected(self):
        for settings in ({"fiscal_years": [2024,2025], "top_n": 1}, {"fiscal_years": [2024,2024]}):
            with self.assertRaises(ValidationError):
                QuestionInput(question="Revenue?", **settings)

    def test_quality_gate_passes_verified_rows_and_rejects_damaged_values(self):
        text = ("CONSOLIDATED STATEMENTS OF OPERATIONS\n(In millions)\n"
                "Total net sales 391,035 383,285 394,328\n"
                "Operating income 123,216 114,301 119,437\nNet income 93,736 96,995 99,803")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"filing.pdf"
            path.write_bytes(b"synthetic PDF fixture")
            page = SimpleNamespace(extract_text=Mock(return_value=text), images=[])
            fake = SimpleNamespace(PdfReader=lambda path: SimpleNamespace(is_encrypted=False, pages=[page]))
            with patch.dict("sys.modules", {"pypdf": fake}):
                parsed = parse_uploaded_pdf(path, 2024, "https://example.com/filing")
                self.assertTrue(all(item["passed"] for item in parsed.quality_checks))
                page.extract_text.return_value = text.replace("391,035", "391")
                with self.assertRaisesRegex(ValueError, "quality gate failed"):
                    parse_uploaded_pdf(path, 2024, "https://example.com/filing")

    def test_database_queries_prefilter_and_parameterize_user_input(self):
        connection = Mock()
        connection.execute.return_value.fetchall.return_value = []
        store = PGStore("not-used")
        request = QuestionInput(question="revenue'); DROP TABLE x; --", fiscal_years=[2024])
        with patch.object(store, "connection", return_value=nullcontext(connection)):
            store.search(request, [0.0]*768, "dataset")
        self.assertEqual(connection.execute.call_count, 2)
        for call in connection.execute.call_args_list:
            sql, parameters = call.args
            self.assertIn("MATERIALIZED", sql)
            self.assertIn("d.fiscal_year=ANY(%s)", sql)
            self.assertNotIn(request.question, sql)
            self.assertEqual(parameters[:4], ["semantic", "AAPL", "dataset", [2024]])

    def test_database_reset_requires_explicit_confirmation(self):
        store = PGStore("not-used")
        with patch.object(store, "connection") as connect:
            with self.assertRaises(ValueError):
                store.reset("yes")
            connect.assert_not_called()

    def row(self, identifier):
        return {"chunk_id": identifier, "document_id": "doc", "pdf_page": 29, "text": "Revenue 120",
                "company": "AAPL", "fiscal_year": 2024, "title": "Apple 10-K", "source_url": "https://example.com",
                "score": 1.0}

    def test_rrf_deduplicates_dense_and_keyword_results(self):
        hits, audit = fuse_candidates([self.row("a"),self.row("b")], [self.row("b"),self.row("a")], 10)
        self.assertEqual(len(hits), 2)
        self.assertEqual({hit.chunk.id for hit in hits}, {"a","b"})
        self.assertTrue(all(item["dense_rank"] and item["keyword_rank"] for item in audit))

    def test_cross_year_selection_reserves_both_filings(self):
        hits = [Hit(Chunk(str(i),str(i),"doc","Revenue", "AAPL",year,"FY","10-K","Results","PDF Page 1","source"),
                    score,"Revenue") for i,(year,score) in enumerate([(2025,1.0),(2025,.9),(2024,.8)])]
        selected = cover_filing_years(hits, [2024,2025], 2)
        self.assertEqual({hit.chunk.fiscal_year for hit in selected}, {2024,2025})

    def test_article_tool_rejects_private_endpoints(self):
        with patch("financial_rag.research_agents.socket.getaddrinfo", return_value=[(None,None,None,None,("127.0.0.1",80))]):
            with self.assertRaisesRegex(ValueError, "private"):
                public_url("http://localhost/article")
        with self.assertRaises(ValueError):
            public_url("file:///etc/passwd")

    def test_captured_evaluation_checks_ids_and_labeled_page_recall(self):
        result = AnswerResult(question="Revenue?", answer="Revenue 120.", citations=[Citation(source_id="E1",
            document_id="doc",document_title="Apple",pdf_page=29)], calculations=[], insufficient_evidence=False,
            status="generated",retrieved_passages={"E1":"Revenue 120."},run_settings={},latency_seconds=.1,
            audit={"generation":{"answer":"Revenue 120.","claims":[{"text":"Revenue 120.","sources":[
                {"evidence_id":"E1","quote":"Revenue 120."}]}],"calculations":[]},
                "context_metadata":{"E1":{"document_id":"doc","pdf_page":29}}})
        checks = deterministic_checks(result, "Revenue 120.", [{"document_id":"doc","pdf_page":29}])
        self.assertTrue(checks["citation_ids_valid"])
        self.assertTrue(checks["grounding_validator_passed"])
        self.assertEqual(checks["recall_at_k"], 1.0)
        self.assertEqual(checks["reciprocal_rank"], 1.0)
        self.assertFalse(checks["numeric_coverage_is_semantic_proof"])
        result.insufficient_evidence = True
        result.audit["generation"]["missing"] = ["Second requested metric unavailable"]
        self.assertTrue(deterministic_checks(result)["grounding_validator_passed"])

    def test_comparison_reuses_candidates_and_retains_all_four_configurations(self):
        """Context: reranking must change ordering without silently running another search."""
        demo = FinancialDemo(Mock(), SimpleNamespace(name="BGE", revision="pinned"), "test-model")
        pools = {strategy: {"hits": [], "candidate_audit": [{"chunk_id": strategy}],
                            "retrieval_seconds": .1} for strategy in ("fixed", "semantic")}
        def generate(request, pool):
            self.assertIs(pool, pools[request.strategy])
            return AnswerResult(question=request.question, answer="Unavailable", citations=[], calculations=[],
                insufficient_evidence=True, status="no_evidence", retrieved_passages={},
                run_settings=request.model_dump(), latency_seconds=.2,
                audit={"generation": {"answer": "Unavailable", "claims": [], "calculations": []}})
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(demo, "retrieve", side_effect=lambda request: pools[request.strategy]) as retrieve:
                with patch.object(demo, "generate", side_effect=generate):
                    rows = demo.compare([{"id": "q1", "question": "Revenue?", "reference": "120"}],
                                        Path(directory)/"comparison.json")
            self.assertEqual(retrieve.call_count, 2)
        self.assertEqual([row["configuration"] for row in rows], ["A", "B", "C", "D"])
        self.assertEqual([row["retrieval_reused"] for row in rows], [False, True, False, True])
        self.assertTrue(all(row["settings"]["candidates"] == 10 and row["settings"]["top_n"] == 5 for row in rows))


class NotebookTests(unittest.TestCase):
    """Context: validate structure and syntax, never execute Colab service/query cells."""

    def test_notebook_cells_are_valid_and_unexecuted(self):
        path = Path(__file__).resolve().parents[1]/"notebooks/Financial_Document_Intelligence_Colab.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        ids = []
        for cell in notebook["cells"]:
            ids.append(cell["id"])
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                ast.parse(cell["source"])
        self.assertEqual(len(ids),len(set(ids)))

    def test_bundled_source_excludes_credentials_and_artifacts(self):
        path = Path(__file__).resolve().parents[1]/"notebooks/Financial_Document_Intelligence_Colab.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cell = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code" and "MODULE_SOURCE =" in cell["source"])
        assignment = next(node for node in ast.parse(cell["source"]).body
                          if isinstance(node,ast.Assign) and getattr(node.targets[0],"id","") == "MODULE_SOURCE")
        sources = ast.literal_eval(assignment.value)
        self.assertTrue(all(name.startswith("financial_rag/") and name.endswith(".py") for name in sources))
        self.assertFalse(any(".env" in name for name in sources))
        for name, source in sources.items():
            ast.parse(source,filename=name)


if __name__ == "__main__":
    unittest.main()

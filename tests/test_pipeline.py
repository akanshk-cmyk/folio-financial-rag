"""Context: these tests target financial failure modes, not just import success.

Fixtures are synthetic. No API keys, external services, or model downloads are used.
"""

from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from financial_rag.calculator import calculate
from financial_rag.chunking import build_index, fixed_chunks, semantic_chunks
from financial_rag.cli import demo
from financial_rag.core import Index, REFUSAL
from financial_rag.generation import answer, validate_response
from financial_rag.ingest import parse_document
from financial_rag.retrieval import retrieve
from financial_rag.evaluation import run_experiments


class FinancialPipelineTests(unittest.TestCase):
    """Context: each test isolates a document-integrity or grounding risk."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        source = self.root / "filing.html"
        source.write_text("""<html><style>NOISE</style><h1 id="results">Results</h1>
<p>USD millions; consolidated GAAP.</p>
<table><tr><th>Metric</th><th>2025</th><th>2024</th></tr>
<tr><td>Revenue</td><td>120</td><td>100</td></tr>
<tr><td>Operating income</td><td>(20)</td><td>15</td></tr></table>
<p>Footnote: revenue includes services.</p><script>INJECTION</script></html>""", encoding="utf-8")
        self.doc = parse_document(source, "DEMO", 2025)

    def tearDown(self):
        self.temporary.cleanup()

    def test_table_labels_values_signs_and_footnotes_survive(self):
        combined = "\n".join(s.text for s in self.doc.sections)
        for value in ("Revenue | 120 | 100", "Operating income | (20) | 15", "Footnote"):
            self.assertIn(value, combined)
        self.assertNotIn("NOISE", combined)
        self.assertNotIn("INJECTION", combined)

    def test_multipart_submission_rejected(self):
        path = self.root / "submission.txt"
        path.write_text("<DOCUMENT><TYPE>10-K</TYPE></DOCUMENT>", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Multipart"):
            parse_document(path, "DEMO", 2025)

    def test_filter_excludes_wrong_company_and_year(self):
        other = replace(self.doc, id="other", ticker="OTHER", fiscal_year=2024)
        index = build_index([self.doc, other], "enhanced")
        hits, pool = retrieve(index, "Revenue", filters={"ticker": "DEMO", "fiscal_year": [2025]}, expand=True)
        self.assertTrue(hits)
        self.assertTrue(all(hit.chunk.ticker == "DEMO" and hit.chunk.fiscal_year == 2025 for hit in hits))
        self.assertIn("Footnote", hits[0].text)

    def test_missing_period_yields_no_evidence(self):
        hits, _ = retrieve(build_index([self.doc], "enhanced"), "Revenue", filters={"fiscal_year": 1900})
        self.assertEqual(answer("Revenue?", hits)["answer"], REFUSAL)

    def test_repeat_build_has_stable_ids_and_roundtrip(self):
        first = build_index([self.doc], "enhanced")
        second = build_index([self.doc], "enhanced")
        self.assertEqual([c.id for c in first.chunks], [c.id for c in second.chunks])
        first.save(self.root / "index.json")
        self.assertEqual(Index.load(self.root / "index.json").parents, first.parents)

    def test_character_split_does_not_drop_digits(self):
        text = "0123456789" * 100
        chunks = list(fixed_chunks(text, 512, 64))
        self.assertEqual(chunks[0][-64:], chunks[1][:64])
        self.assertEqual(chunks[0] + "".join(c[64:] for c in chunks[1:]), text)

    def test_semantic_needs_real_embedder(self):
        with self.assertRaisesRegex(ValueError, "requires model"):
            build_index([self.doc], "semantic")

    def test_percentile_setting_changes_semantic_boundaries(self):
        # Context: known sentence distances should produce more boundaries at
        # percentile 65 than 95; no model download is needed to test the splitter.
        from math import acos, cos, sin
        class FakeEmbedder:
            def encode(self, texts):
                angles = [0.0]
                for distance in [0.0, .2, .4, .6, .8, .95]:
                    angles.append(angles[-1] + acos(1 - distance))
                return [[cos(angle), sin(angle)] for angle in angles]
        text = "A. B. C. D. E. F. G."
        lower = semantic_chunks(text, FakeEmbedder(), 65)
        higher = semantic_chunks(text, FakeEmbedder(), 95)
        self.assertEqual(len(lower), 3)
        self.assertEqual(len(higher), 2)
        self.assertEqual(" ".join(lower), text)

    def test_semantic_boundary_retains_sentences(self):
        class FakeEmbedder:
            def encode(self, texts):
                return [[1, 0], [1, 0], [0, 1], [0, 1]]
        result = semantic_chunks("One. Two. Three. Four.", FakeEmbedder())
        self.assertEqual(" ".join(result), "One. Two. Three. Four.")
        self.assertGreater(len(result), 1)

    def test_rerank_receives_identical_saved_pool(self):
        index = build_index([self.doc], "fixed")
        _, pool = retrieve(index, "Revenue", candidates=10, top_n=1)
        with patch("financial_rag.retrieval.flashrank", side_effect=lambda hits, query: list(reversed(hits))) as reranker:
            _, replayed = retrieve(index, "Revenue", candidates=10, top_n=1, rerank=True, saved_candidates=pool)
        self.assertEqual(pool, replayed)
        self.assertEqual([h.chunk.id for h in reranker.call_args.args[0]], [p["id"] for p in pool])

    def test_fabricated_quote_or_number_is_rejected(self):
        payload = {"answerable": True, "claims": [{"text": "Revenue was 999.",
                   "sources": [{"evidence_id": "E1", "quote": "Revenue was 120."}]}]}
        with self.assertRaisesRegex(ValueError, "numerals"):
            validate_response(payload, {"E1": "Revenue was 120."})
        payload["claims"][0]["sources"][0]["quote"] = "Revenue was 999."
        with self.assertRaisesRegex(ValueError, "invalid source"):
            validate_response(payload, {"E1": "Revenue was 120."})

    def operands(self):
        common = {"currency": "USD", "scale": "millions", "basis": "GAAP", "scope": "consolidated",
                  "period_type": "annual", "metric": "revenue", "evidence_id": "E1",
                  "quote": "FY2025 revenue 120; FY2024 revenue 100."}
        return [{**common, "value": "120", "period": "FY2025"},
                {**common, "value": "100", "period": "FY2024"}]

    def test_calculator_uses_cited_comparable_inputs(self):
        inputs = self.operands()
        result = calculate("growth", inputs, {"E1": inputs[0]["quote"]})
        self.assertEqual(Decimal(result["raw_output"]), Decimal("20"))
        self.assertEqual(result["display_output"], "20.00")
        inputs[1]["scale"] = "billions"
        with self.assertRaisesRegex(ValueError, "scale"):
            calculate("growth", inputs, {"E1": inputs[0]["quote"]})

    def test_zero_denominator_rejected(self):
        inputs = self.operands()
        inputs[1]["value"] = "0"
        for item in inputs:
            item["quote"] = "FY2025 revenue 120; FY2024 revenue 0."
        with self.assertRaisesRegex(ValueError, "zero"):
            calculate("growth", inputs, {"E1": inputs[0]["quote"]})

    def test_budget_drops_whole_evidence_without_truncating(self):
        hits, _ = retrieve(build_index([self.doc], "enhanced"), "Revenue", expand=True)
        result = answer("Revenue?", hits, max_context_chars=1)
        self.assertEqual(result["answer"], REFUSAL)
        self.assertTrue(result["dropped_chunks"])

    def test_offline_demo_is_explicitly_synthetic(self):
        with patch("builtins.print"):
            demo(str(self.root / "demo"))
        result = json.loads((self.root / "demo" / "answer.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "preview-only")
        self.assertTrue(all(s["synthetic"] for s in result["sources"].values()))

    def test_invalid_model_response_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "JSON object"):
            validate_response([], {})

    def test_unknown_calculation_units_are_rejected(self):
        inputs = self.operands()
        for item in inputs:
            item["scale"] = "unknown"
        with self.assertRaisesRegex(ValueError, "explicit"):
            calculate("growth", inputs, {"E1": inputs[0]["quote"]})

    def test_hosted_answer_adapter_without_network(self):
        hits, _ = retrieve(build_index([self.doc], "enhanced"), "Revenue", expand=True)
        payload = {"answerable": True, "claims": [{"text": "Revenue includes 120.",
                   "sources": [{"evidence_id": "E1", "quote": "Revenue | 120 | 100"}]}],
                   "calculations": [], "missing": []}
        response = SimpleNamespace(id="mock-response", usage=None,
                                   choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(payload)))])
        create = unittest.mock.Mock(return_value=response)
        fake = SimpleNamespace(APIError=RuntimeError,
                               Groq=lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
        with patch.dict("sys.modules", {"groq": fake}), patch.dict("os.environ", {"GROQ_API_KEY": "test-only"}):
            result = answer("Revenue?", hits, "groq", "test-model")
        self.assertEqual(result["status"], "generated")
        self.assertIn(hits[0].chunk.citation(), result["answer"])
        self.assertEqual(create.call_args.kwargs["n"], 1)
        self.assertEqual(create.call_args.kwargs["response_format"], {"type": "json_object"})
        self.assertIn("retrieved_context", create.call_args.kwargs["messages"][1]["content"])

    def test_experiment_reuses_semantic_candidates_and_logs_manifest(self):
        class FakeEmbedder:
            name = "test-model"
            revision = "test-revision"
            def encode(self, texts, query=False):
                return [[1.0, .1] for _ in texts]
        model = FakeEmbedder()
        fixed = build_index([self.doc], "fixed", model)
        semantic = build_index([self.doc], "semantic", model)
        questions = [{"id": "q1", "question": "Revenue?", "reference": "Synthetic revenue 120."}]
        with patch("financial_rag.retrieval.flashrank", side_effect=lambda hits, query: hits):
            rows = run_experiments(fixed, semantic, questions, model, self.root / "experiment.json")
        self.assertEqual([row["configuration"] for row in rows], ["A", "B", "C"])
        self.assertEqual(rows[1]["candidate_pool"], rows[2]["candidate_pool"])
        result = json.loads((self.root / "experiment.json").read_text(encoding="utf-8"))
        self.assertIn("prompt_hash", result["manifest"])
        # Context: a completed benchmark resumes without changing answers or making calls.
        with patch("financial_rag.evaluation.build_rag_chain") as chain:
            resumed=run_experiments(fixed,semantic,questions,model,self.root/"experiment.json",resume=True)
        chain.assert_not_called()
        self.assertEqual(resumed,rows)
        with self.assertRaisesRegex(ValueError,"changed settings"):
            run_experiments(fixed,semantic,[{**questions[0],"reference":"Changed reference"}],model,
                            self.root/"experiment.json",resume=True)

    def test_reference_template_is_rejected_before_model_calls(self):
        with self.assertRaisesRegex(ValueError, "manually verified"):
            run_experiments(None, None, [{"reference": "REPLACE WITH AN ANSWER"}],
                            None, self.root / "run.json")


if __name__ == "__main__":
    unittest.main()

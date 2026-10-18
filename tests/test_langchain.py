"""Context: integration tests exercise real LCEL and splitters without live APIs.

Known embeddings isolate boundary behavior; a mock transport isolates generation.
Financial evidence checks still use the production validation code.
"""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import json

from langchain_core.runnables import RunnableSequence

from financial_rag.chains import build_rag_chain
from financial_rag.chunking import build_index, fixed_chunks, semantic_chunks, CHUNKING_IMPLEMENTATION
from financial_rag.core import Document, Section
from financial_rag.embedding import LangChainEmbeddings
from financial_rag.generation import answer, build_generation_chain, SYSTEM


class LangChainIntegrationTests(unittest.TestCase):
    """Context: check source scope, budget, strict grounding, and literal prompt values."""

    def index(self):
        section = Section("section", "Revenue was 120.\nFootnote: USD millions.", "Results", "PDF Page 1")
        documents = [Document("demo", "DEMO", 2025, "FY", "Synthetic", "synthetic", "hash", [section], synthetic=True),
                     Document("other", "OTHER", 2024, "FY", "Synthetic", "synthetic", "hash2", [section], synthetic=True)]
        return build_index(documents, "enhanced")

    def test_rag_chain_preserves_filters_and_sources(self):
        chain = build_rag_chain(self.index(), filters={"ticker": "DEMO", "fiscal_year": 2025}, expand=True)
        self.assertIsInstance(chain, RunnableSequence)
        result = chain.invoke({"question": "Revenue?"})
        self.assertEqual(result["status"], "preview-only")
        self.assertEqual(result["orchestration"], "langchain-lcel-v1")
        self.assertTrue(result["candidate_pool"])
        self.assertTrue(all("demo," in source["citation"] for source in result["sources"].values()))
        self.assertIn("Footnote", next(iter(result["evidence"].values())))

    def test_empty_question_is_rejected_before_retrieval(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            build_rag_chain(self.index()).invoke({"question": " "})

    def test_missing_evidence_never_calls_groq(self):
        with patch("financial_rag.generation.request_groq") as transport:
            result = build_rag_chain(self.index(), filters={"fiscal_year": 1900}, provider="groq").invoke(
                {"question": "Revenue?"})
        self.assertEqual(result["status"], "no-evidence")
        transport.assert_not_called()

    def test_lcel_calculation_keeps_formula_sources_and_output(self):
        quote = "FY2025 revenue 120; FY2024 revenue 100."
        common = {"currency": "USD", "scale": "millions", "basis": "GAAP", "scope": "consolidated",
                  "period_type": "annual", "metric": "revenue", "evidence_id": "E1", "quote": quote}
        payload = {"answerable": True, "claims": [], "missing": [], "calculations": [{"operation": "growth",
            "inputs": [{**common, "period": "FY2025", "value": "120"},
                       {**common, "period": "FY2024", "value": "100"}]}]}
        response = SimpleNamespace(id="fake", usage=None, choices=[SimpleNamespace(finish_reason="stop",
                                   message=SimpleNamespace(content=json.dumps(payload)))])
        base = {"evidence": {"E1": quote}, "sources": {"E1": {"citation": "[demo, Results, PDF Page 1]"}}}
        with patch("financial_rag.generation.request_groq", return_value=response):
            result = build_generation_chain("mock-model").invoke({"base": base, "evidence": base["evidence"],
                                                                   "question": "Revenue growth?"})
        self.assertEqual(result["status"], "generated")
        self.assertEqual(result["calculations"][0]["display_output"], "20.00")
        self.assertIn("[demo, Results, PDF Page 1]", result["answer"])

    def test_huggingface_adapter_distinguishes_query_from_documents(self):
        encoder = SimpleNamespace(encode=Mock(return_value=[[1.0, 0.0]]))
        adapter = LangChainEmbeddings(encoder)
        self.assertEqual(adapter.embed_documents(["passage"]), [[1.0, 0.0]])
        encoder.encode.assert_called_with(["passage"])
        self.assertEqual(adapter.embed_query("question"), [1.0, 0.0])
        encoder.encode.assert_called_with(["question"], query=True)

    def test_fixed_splitter_bounds_and_overlap_on_unbroken_text(self):
        text = "X" * 1200
        pieces = fixed_chunks(text)
        self.assertTrue(all(len(piece) <= 512 for piece in pieces))
        self.assertEqual(sum(len(piece) for piece in pieces) - 64 * (len(pieces) - 1), len(text))

    def test_semantic_secondary_limit_preserves_long_single_sentence(self):
        class NoCallEmbedder:
            def encode(self, texts):
                raise AssertionError("A single sentence should not need semantic-distance embedding.")
        text = "Z" * 17000
        groups = semantic_chunks(text, NoCallEmbedder())
        self.assertTrue(all(len(group) <= 8000 for group in groups))
        self.assertEqual("".join(groups), text)

    def test_manifest_identifies_langchain_implementation(self):
        manifest = self.index().manifest
        self.assertEqual(manifest["chunking_implementation"], CHUNKING_IMPLEMENTATION)
        self.assertEqual(manifest["schema_version"], 2)
        self.assertTrue(manifest["langchain_text_splitters_version"])

    def test_prompt_preserves_json_braces_and_rejects_invented_number(self):
        # Context: document/query braces must remain literal data, while an
        # unsupported generated value must still fail validation after LCEL.
        evidence = {"E1": 'Revenue was 120. Data: {"note":"untrusted"}'}
        base = {"evidence": evidence, "sources": {"E1": {"citation": "[demo, Results, PDF Page 1]"}}}
        payload = {"answerable": True, "claims": [{"text": "Revenue was 999.", "sources": [
            {"evidence_id": "E1", "quote": "Revenue was 120."}]}], "calculations": [], "missing": []}
        response = SimpleNamespace(id="fake", usage=None, choices=[SimpleNamespace(finish_reason="stop",
                                   message=SimpleNamespace(content=json.dumps(payload)))])
        with patch("financial_rag.generation.request_groq", return_value=response) as transport:
            result = build_generation_chain("mock-model").invoke({"base": base, "evidence": evidence,
                                                                   "question": 'What is {"revenue"}?'})
        messages = transport.call_args.args[0].to_messages()
        self.assertEqual(messages[0].content, SYSTEM)
        self.assertEqual(json.loads(messages[1].content)["retrieved_context"], evidence)
        self.assertEqual(result["status"], "validation-failed")
        self.assertIn("numerals", result["validation"])


if __name__ == "__main__":
    unittest.main()

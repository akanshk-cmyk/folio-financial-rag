"""Context: provider migration checks never send financial data or make live calls.

SDK and RAGAS checks run when their optional extras are installed. The remaining
checks run offline and ensure OpenAI credentials cannot be used as a fallback.
"""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from financial_rag.cli import parser
from financial_rag.core import Chunk
from financial_rag.evaluation import evaluate_ragas
from financial_rag.generation import answer, resolve_model, groq_error_message
from financial_rag.retrieval import Hit


class GroqConfigurationTests(unittest.TestCase):
    """Context: test explicit provider/model selection independently of installed SDKs."""

    def hit(self):
        chunk = Chunk("child", "parent", "DEMO", "Revenue was 120.", "DEMO", 2025,
                      "FY", "Synthetic", "Results", "PDF Page 1", "synthetic")
        return Hit(chunk, 1, chunk.text)

    def test_model_overrides_and_judge_setting(self):
        with patch.dict(os.environ, {"GROQ_MODEL": "generator", "GROQ_JUDGE_MODEL": "judge"}, clear=True):
            self.assertEqual(resolve_model(), "generator")
            self.assertEqual(resolve_model(judge=True), "judge")
            self.assertEqual(resolve_model("explicit", judge=True), "explicit")

    def test_404_reports_model_access_without_echoing_provider_message(self):
        error = SimpleNamespace(status_code=404, body={"error": {
            "code": "model_not_found", "message": "sensitive content must not be echoed"}})
        message = groq_error_message(error, "test-model")
        self.assertIn("model_not_found", message)
        self.assertIn("unavailable", message)
        self.assertNotIn("sensitive", message)
        self.assertNotIn("rate limit", message)

    def test_429_is_identified_as_rate_limit(self):
        error = SimpleNamespace(status_code=429, body={"error": {"code": "rate_limit_exceeded"}})
        self.assertIn("quota/rate limit", groq_error_message(error, "test-model"))

    def test_cli_accepts_groq_without_overriding_environment_model(self):
        args = parser().parse_args(["ask", "Revenue?", "--index", "unused.json", "--provider", "groq"])
        self.assertEqual(args.provider, "groq")
        self.assertIsNone(args.model)

    def test_openai_key_is_not_a_groq_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "unused-test-value"}, clear=True), \
                patch("financial_rag.generation.load_environment"):
            with self.assertRaisesRegex(ValueError, "GROQ_API_KEY"):
                answer("Revenue?", [self.hit()], "groq")

    @unittest.skipUnless(importlib.util.find_spec("groq"), "Install cloud extras")
    def test_real_sdk_request_and_response_using_mock_transport(self):
        import groq
        import httpx
        captured = []
        payload = {"answerable": True, "claims": [{"text": "Revenue was 120.",
                   "sources": [{"evidence_id": "E1", "quote": "Revenue was 120."}]}],
                   "calculations": [], "missing": []}

        def handle(request):
            captured.append(json.loads(request.content))
            self.assertEqual(request.url.host, "api.groq.com")
            self.assertEqual(request.url.path, "/openai/v1/chat/completions")
            return httpx.Response(200, json={"id": "mock", "object": "chat.completion", "created": 1,
                "model": "mock-model", "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": json.dumps(payload)}}]})

        with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
            client = groq.Groq(api_key="test-only", http_client=http_client)
            with patch("groq.Groq", return_value=client), \
                    patch.dict(os.environ, {"GROQ_API_KEY": "test-only"}), \
                    patch("financial_rag.generation.load_environment"):
                result = answer("Revenue?", [self.hit()], "groq", "openai/gpt-oss-120b")
        self.assertEqual(result["status"], "generated")
        self.assertEqual(captured[0]["n"], 1)
        self.assertEqual(captured[0]["response_format"], {"type": "json_object"})
        self.assertEqual(captured[0]["reasoning_effort"], "low")

    @unittest.skipUnless(importlib.util.find_spec("ragas") and importlib.util.find_spec("langchain_groq"),
                         "Install evaluation extras")
    def test_ragas_uses_groq_judge_without_embedding_api(self):
        from ragas.metrics.base import MetricWithEmbeddings
        with tempfile.TemporaryDirectory() as folder:
            source, destination = Path(folder) / "run.json", Path(folder) / "evaluation.json"
            source.write_text(json.dumps({"manifest": {}, "rows": [{"status": "generated",
                "question": "Revenue?", "answer": "Revenue was 120.", "reference": "120",
                "evidence": {"E1": "Revenue was 120."}, "configuration": "A", "question_id": "q1"}]}))
            fake = SimpleNamespace(to_pandas=lambda: SimpleNamespace(to_dict=lambda **kwargs: [{"faithfulness": 1.0}]))
            with patch.dict(os.environ, {"GROQ_API_KEY": "test-only"}, clear=True), \
                    patch("financial_rag.evaluation.load_environment"), \
                    patch("ragas.evaluate", return_value=fake) as evaluate:
                evaluate_ragas(source, destination, "openai/gpt-oss-120b", provider="groq")
            settings = evaluate.call_args.kwargs
            self.assertTrue(settings["llm"].bypass_n)
            self.assertEqual(settings["llm"].langchain_llm.model_name, "openai/gpt-oss-120b")
            self.assertEqual(settings["llm"].langchain_llm.reasoning_effort, "low")
            self.assertEqual(settings["run_config"].max_workers, 1)
            self.assertNotIn("embeddings", settings)
            self.assertTrue(all(not isinstance(metric, MetricWithEmbeddings) for metric in settings["metrics"]))
            self.assertEqual(json.loads(destination.read_text())["judge_provider"], "groq")


if __name__ == "__main__":
    unittest.main()

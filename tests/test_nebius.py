"""Context: verify Nebius transport, adaptive judging and RAGAS without API calls."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
import openai

from financial_rag.core import Chunk
from financial_rag.generation import answer, resolve_model, nebius_base_url
from financial_rag.retrieval import Hit
from financial_rag.evaluation import evaluate_ragas
from financial_rag.adaptive import judge_json
from financial_rag.evidence_agent import planner_from_environment


class NebiusProviderTests(unittest.TestCase):
    def test_settings_and_credentials_are_isolated(self):
        # Context: another provider's credentials never satisfy Nebius authentication.
        with patch.dict(os.environ, {"OPENAI_API_KEY":"other", "GROQ_MODEL":"other",
                "NEBIUS_MODEL":"generator", "NEBIUS_JUDGE_MODEL":"judge"}, clear=True):
            self.assertEqual(resolve_model(provider="nebius"), "generator")
            self.assertEqual(resolve_model(judge=True, provider="nebius"), "judge")
            self.assertEqual(resolve_model("explicit", provider="nebius"), "explicit")
            chunk = Chunk("c","p","d","Revenue 120.","DEMO",2025,"FY","10-K","Results","PDF Page 1","local")
            with patch("financial_rag.generation.load_environment"):
                with self.assertRaisesRegex(ValueError,"NEBIUS_API_KEY"):
                    answer("Revenue?", [Hit(chunk,1,chunk.text)], "nebius")

    def test_sdk_sends_nebius_key_endpoint_and_grounded_json(self):
        # Context: exercise the installed SDK with a fake HTTP transport and real validator.
        payload={"answerable":True,"claims":[{"text":"Revenue 120.","sources":[
            {"evidence_id":"E1","quote":"Revenue 120."}]}],"calculations":[],"missing":[]}
        captured=[]
        def handle(request):
            self.assertEqual(request.url.host,"api.tokenfactory.nebius.com")
            self.assertEqual(request.url.path,"/v1/chat/completions")
            self.assertEqual(request.headers["authorization"],"Bearer nebius-test")
            captured.append(json.loads(request.content))
            return httpx.Response(200,json={"id":"mock","object":"chat.completion","created":1,
                "model":"test-model","choices":[{"index":0,"finish_reason":"stop",
                "message":{"role":"assistant","content":json.dumps(payload)}}]})
        client=openai.OpenAI(api_key="nebius-test",base_url="https://api.tokenfactory.nebius.com/v1/",
            http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        chunk=Chunk("c","p","d","Revenue 120.","DEMO",2025,"FY","10-K","Results","PDF Page 1","local")
        with patch.dict(os.environ,{"NEBIUS_API_KEY":"nebius-test"},clear=True), patch("financial_rag.generation.load_environment"):
            with patch("openai.OpenAI",return_value=client) as factory:
                result=answer("Revenue?",[Hit(chunk,1,chunk.text)],"nebius", "test-model")
            self.assertEqual(factory.call_args.kwargs["base_url"],nebius_base_url())
        self.assertEqual(result["status"],"generated")
        self.assertEqual(result["provider"],"nebius")
        self.assertEqual(captured[0]["max_tokens"],3000)
        self.assertEqual(captured[0]["n"],1)
        self.assertEqual(captured[0]["response_format"],{"type":"json_object"})

    def test_adaptive_checks_use_nebius(self):
        # Context: graders and checkers must use the same selected service as generation.
        response=SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
            message=SimpleNamespace(content='{"grounded":true}'))])
        with patch("financial_rag.adaptive.request_nebius",return_value=response) as request:
            self.assertTrue(judge_json("Check",{},"nebius","test")["grounded"])
        self.assertEqual(request.call_args.kwargs["max_completion_tokens"],600)

    def test_ragas_uses_nebius_without_embedding_api(self):
        # Context: all three metrics use a single-completion Nebius judge and no remote embeddings.
        with tempfile.TemporaryDirectory() as folder:
            source, target=Path(folder)/"run.json", Path(folder)/"scores.json"
            source.write_text(json.dumps({"manifest":{},"rows":[{"status":"generated",
                "question":"Revenue?","answer":"120","evidence":{"E1":"Revenue 120"},
                "reference":"120","configuration":"A","question_id":"Q1"}]}))
            fake=SimpleNamespace(to_pandas=lambda:SimpleNamespace(to_dict=lambda **kw:[{"faithfulness":1.0}]))
            with patch.dict(os.environ,{"NEBIUS_API_KEY":"test","NEBIUS_JUDGE_MODEL":"judge"}):
                with patch("financial_rag.evaluation.load_environment"), patch("ragas.evaluate",return_value=fake) as evaluate:
                    evaluate_ragas(source,target,provider="nebius")
                judge=evaluate.call_args.kwargs["llm"].langchain_llm
                self.assertEqual(judge.model_name,"judge")
                self.assertEqual(judge.openai_api_base,nebius_base_url())
                self.assertEqual(judge.n,1)
                self.assertEqual(judge.model_kwargs["response_format"],{"type":"json_object"})
                self.assertEqual(len(evaluate.call_args.kwargs["metrics"]),3)
                self.assertNotIn("embeddings",evaluate.call_args.kwargs)
                self.assertEqual(json.loads(target.read_text())["judge_provider"],"nebius")

    def test_research_planner_uses_nebius(self):
        # Context: the Streamlit provider also controls the tool-calling research page.
        with patch.dict(os.environ,{"NEBIUS_API_KEY":"test"}):
            planner=planner_from_environment("nebius","test-model")
            self.assertEqual(planner.openai_api_base,nebius_base_url())
            self.assertEqual(planner.model_name,"test-model")


if __name__ == "__main__":
    unittest.main()

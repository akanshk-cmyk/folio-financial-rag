"""Context: verify provider isolation and request schemas without live API calls."""

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
from financial_rag.evaluation import evaluate_ragas
from financial_rag.generation import answer,resolve_model
from financial_rag.retrieval import Hit


class OpenAIProviderTests(unittest.TestCase):
    def hit(self):
        chunk=Chunk("c","p","DEMO","Revenue was 120.","DEMO",2025,"FY","Synthetic","Results","PDF Page 1","synthetic")
        return Hit(chunk,1,chunk.text)

    def test_openai_uses_its_own_model_settings(self):
        with patch.dict(os.environ,{"GROQ_MODEL":"groq-model","OPENAI_MODEL":"openai-model","OPENAI_JUDGE_MODEL":"openai-judge"},clear=True):
            self.assertEqual(resolve_model(provider="openai"),"openai-model")
            self.assertEqual(resolve_model(judge=True,provider="openai"),"openai-judge")
            self.assertEqual(resolve_model(),"groq-model")

    def test_groq_key_is_never_used_for_openai(self):
        with patch.dict(os.environ,{"GROQ_API_KEY":"test-only"},clear=True),patch("financial_rag.generation.load_environment"):
            with self.assertRaisesRegex(ValueError,"OPENAI_API_KEY"):
                answer("Revenue?",[self.hit()],"openai")

    def test_real_sdk_transport_uses_openai_endpoint_and_same_grounding(self):
        payload={"answerable":True,"claims":[{"text":"Revenue was 120.","sources":[
            {"evidence_id":"E1","quote":"Revenue was 120."}]}],"calculations":[],"missing":[]}
        captured=[]
        def handle(request):
            self.assertEqual(request.url.host,"api.openai.com")
            self.assertEqual(request.url.path,"/v1/chat/completions")
            captured.append(json.loads(request.content))
            return httpx.Response(200,json={"id":"mock","object":"chat.completion","created":1,
                "model":"gpt-4.1-mini-2025-04-14","choices":[{"index":0,"finish_reason":"stop",
                    "message":{"role":"assistant","content":json.dumps(payload)}}]})
        client=openai.OpenAI(api_key="test-only",http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        with patch("openai.OpenAI",return_value=client),patch.dict(os.environ,{"OPENAI_API_KEY":"test-only"}),patch("financial_rag.generation.load_environment"):
            result=answer("Revenue?",[self.hit()],"openai")
        self.assertEqual(result["provider"],"openai")
        self.assertEqual(result["status"],"generated")
        self.assertEqual(captured[0]["response_format"],{"type":"json_object"})
        self.assertEqual(captured[0]["n"],1)
        self.assertNotIn("reasoning_effort",captured[0])

    def test_ragas_uses_explicit_openai_judge_without_embedding_api(self):
        with tempfile.TemporaryDirectory() as folder:
            source,destination=Path(folder)/"input.json",Path(folder)/"results.json"
            source.write_text(json.dumps({"manifest":{},"rows":[{"status":"generated","question":"Revenue?",
                "answer":"Revenue 120.","evidence":{"E1":"Revenue 120."},"reference":"Revenue 120.",
                "configuration":"OpenAI","question_id":"Q1"}]}))
            fake=SimpleNamespace(to_pandas=lambda:SimpleNamespace(to_dict=lambda **kwargs:[{"faithfulness":1.0}]))
            with patch.dict(os.environ,{"OPENAI_API_KEY":"test-only"},clear=True),patch("financial_rag.evaluation.load_environment"),patch("ragas.evaluate",return_value=fake) as evaluate:
                evaluate_ragas(source,destination,"gpt-4.1-mini-2025-04-14",provider="openai")
            self.assertEqual(evaluate.call_args.kwargs["llm"].langchain_llm.model_name,"gpt-4.1-mini-2025-04-14")
            self.assertNotIn("embeddings",evaluate.call_args.kwargs)
            self.assertEqual(json.loads(destination.read_text())["judge_provider"],"openai")


if __name__=="__main__":
    unittest.main()

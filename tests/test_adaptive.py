"""Context: exercise adaptive retries and fail-closed decisions without live APIs."""

import unittest
from unittest.mock import Mock, patch

from financial_rag.adaptive import adaptive_answer
from financial_rag.core import Chunk, Index, REFUSAL
from financial_rag.retrieval import Hit


class AdaptiveTests(unittest.TestCase):
    def setUp(self):
        self.chunk=Chunk("c","p","A","2025 2024\nRevenue 120 100","AAA",2025,"FY","10-K","Income","Page 3","source")
        self.index=Index({"strategy":"enhanced"},[self.chunk],{"p":self.chunk.text})
        self.search=Mock(return_value=([Hit(self.chunk,1,self.chunk.text)],[{"id":"c","score":1}]))
        self.result={"question":"Revenue?","answer":"Revenue was 120.","status":"generated",
            "claims":[{"text":"Revenue was 120.","sources":[{"evidence_id":"E1","quote":"Revenue 120 100"}]}],
            "evidence":{"E1":self.chunk.text},"sources":{"E1":{"citation":"[A, Page 3]"}},"calculations":[],"missing":[]}

    def run_answer(self,judge):
        return adaptive_answer(self.index,"Revenue in 2025?",ticker="AAA",years=[2025],provider="groq",
                               search_backend=self.search,namespace="semantic",judge=judge)

    def test_broader_search_keeps_scope_then_returns_checked_answer(self):
        judge=Mock(side_effect=[{"relevant_ids":[],"sufficient":False},
                               {"relevant_ids":["R1"],"sufficient":True},{"grounded":True}])
        with patch("financial_rag.adaptive.answer",return_value=self.result) as generate:
            result=self.run_answer(judge)
        self.assertEqual(result["status"],"generated")
        self.assertEqual(self.search.call_count,2)
        self.assertEqual([call.args[3] for call in self.search.call_args_list],
                         [{"ticker":"AAA","fiscal_year":[2025]}]*2)
        self.assertGreater(self.search.call_args_list[1].args[4],self.search.call_args_list[0].args[4])
        self.assertEqual(generate.call_args.kwargs["max_retries"],0)
        self.assertEqual(result["route"]["namespace"],"semantic")

    def test_irrelevant_evidence_never_reaches_generator(self):
        judge=Mock(return_value={"relevant_ids":[],"sufficient":False})
        with patch("financial_rag.adaptive.answer") as generate:
            result=self.run_answer(judge)
        generate.assert_not_called()
        self.assertEqual(result["answer"],REFUSAL)
        self.assertEqual(len(result["retrieval_attempts"]),2)

    def test_failed_grounding_caps_generation_at_three_and_withholds_answer(self):
        judge=Mock(side_effect=[{"relevant_ids":["R1"],"sufficient":True}]+[{"grounded":False,"reason":"Wrong year"}]*3)
        with patch("financial_rag.adaptive.answer",return_value=self.result) as generate:
            result=self.run_answer(judge)
        self.assertEqual(generate.call_count,3)
        self.assertEqual(result["status"],"grounding-failed")
        self.assertEqual(result["answer"],REFUSAL)
        self.assertIn("Wrong year",generate.call_args.args[0])

    def test_generator_refusal_triggers_retrieval_retry(self):
        judge=Mock(side_effect=[{"relevant_ids":["R1"],"sufficient":True}]*2+[{"grounded":True}])
        with patch("financial_rag.adaptive.answer",side_effect=[{**self.result,"answer":REFUSAL,"missing":["Revenue"]},self.result]):
            result=self.run_answer(judge)
        self.assertEqual(self.search.call_count,2)
        self.assertEqual(result["status"],"generated")

    def test_untrusted_grader_ids_and_out_of_scope_retrieval_are_rejected(self):
        with self.assertRaisesRegex(ValueError,"invalid source"):
            self.run_answer(Mock(return_value={"relevant_ids":["fabricated"],"sufficient":True}))
        foreign=Chunk("x","x","B","Revenue 999","BBB",2025,"FY","10-K","Income","Page 1","source")
        self.search.return_value=([Hit(foreign,1,foreign.text)],[])
        with self.assertRaisesRegex(ValueError,"outside"):
            self.run_answer(Mock())


if __name__=="__main__":
    unittest.main()

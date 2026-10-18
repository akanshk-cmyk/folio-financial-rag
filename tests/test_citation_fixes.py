"""Context: citation tolerance must preserve factual and year boundaries."""

import unittest
import json
from types import SimpleNamespace
from unittest.mock import patch

from financial_rag.calculator import canonical_quote
from financial_rag.core import Chunk
from financial_rag.generation import validate_response, build_generation_chain
from financial_rag.retrieval import bm25, financial_row_label


class CitationFixTests(unittest.TestCase):
    def payload(self, text, quotes):
        return {"answerable":True,"claims":[{"text":text,"sources":[
            {"evidence_id":"E1","quote":quote} for quote in quotes]}],"calculations":[],"missing":[]}

    def test_pdf_whitespace_returns_original_span(self):
        original="Supply shortages can\n   disrupt manufacturing."
        self.assertEqual(canonical_quote("Supply shortages can disrupt manufacturing.",original),original)

    def test_ellipses_and_altered_financial_values_remain_invalid(self):
        for quote in ["Revenue 120...", "Revenue 999.", "Revenue 1 20."]:
            with self.assertRaises(ValueError):
                canonical_quote(quote,"Revenue 120.")

    def test_header_and_metric_quotes_jointly_support_year_and_value(self):
        evidence={"E1":"Annual results (USD millions)\n2025 2024\nRevenue 120 100"}
        payload=self.payload("FY2025 revenue was 120 million.",["2025 2024","Revenue 120 100"])
        self.assertEqual(validate_response(payload,evidence),[])

    def test_wrong_year_or_year_outside_a_table_header_remains_invalid(self):
        evidence={"E1":"2025 2024\nRevenue 120 100"}
        for text,quotes in [("FY2023 revenue was 120.",["2025 2024","Revenue 120 100"])]:
            with self.assertRaisesRegex(ValueError,"numerals"):
                validate_response(self.payload(text,quotes),evidence)
        with self.assertRaisesRegex(ValueError,"numerals"):
            validate_response(self.payload("FY2025 revenue was 120.",["Revenue 120 100"]),
                              {"E1":"FY2025 filing\nRevenue 120 100"})

    def test_verified_table_row_is_expanded_to_original_year_header(self):
        evidence={"E1":"2025 2024\nRevenue 120 100"}
        payload=self.payload("FY2025 revenue was 120.",["Revenue 120 100"])
        validate_response(payload,evidence)
        source=payload["claims"][0]["sources"][0]
        self.assertEqual(source["quote"],evidence["E1"])
        self.assertEqual(source["original_quote"],"Revenue 120 100")

    def test_table_header_recovery_never_skips_a_conflicting_nearer_header(self):
        evidence={"E1":"2023 2022\nOther 80 70\n2025 2024\nRevenue 120 100"}
        with self.assertRaisesRegex(ValueError,"numerals"):
            validate_response(self.payload("FY2023 revenue was 120.",["Revenue 120 100"]),evidence)

    def test_year_shaped_amount_cannot_be_recovered_from_header(self):
        with self.assertRaisesRegex(ValueError,"numerals"):
            validate_response(self.payload("Revenue was 2025 million.",["Revenue 120 100"]),
                              {"E1":"2025 2024\nRevenue 120 100"})

    def test_narrative_validation_keeps_literal_source_span(self):
        original="Suppliers can\n  delay production."
        payload=self.payload("Supplier delays are a production risk.",["Suppliers can delay production."])
        validate_response(payload,{"E1":original})
        self.assertEqual(payload["claims"][0]["sources"][0]["quote"],original)

    def test_invalid_duplicate_pointer_is_removed_only_with_identical_verified_quote(self):
        payload=self.payload("Revenue was 120.",["Revenue 120."])
        payload["claims"][0]["sources"].append({"evidence_id":"E2","quote":"Revenue   120."})
        validate_response(payload,{"E1":"Revenue 120.","E2":"Unrelated text"})
        claim=payload["claims"][0]
        self.assertEqual(len(claim["sources"]),1)
        self.assertEqual(claim["discarded_duplicate_sources"][0]["evidence_id"],"E2")
        bad=self.payload("Revenue was 120.",["Revenue 120."])
        bad["claims"][0]["sources"].append({"evidence_id":"E2","quote":"Revenue 999."})
        with self.assertRaisesRegex(ValueError,"invalid source"):
            validate_response(bad,{"E1":"Revenue 120.","E2":"Unrelated text"})

    def test_numeric_statement_row_beats_incidental_metric_mentions(self):
        def chunk(identifier,text):
            return Chunk(identifier,identifier,"doc",text,"TEST",2025,"FY","10-K","Statement","PDF Page 1","source")
        statement=chunk("statement","2025 2024\nCash and cash equivalents 120 100")
        prose=chunk("prose","Cash and cash equivalents are discussed. "*20)
        self.assertEqual(bm25("What was cash and cash equivalents?",[prose,statement])[0].chunk.id,"statement")

    def test_revenue_question_finds_equivalent_statement_labels(self):
        # Context: the UI says revenues, while issuers use differing sales labels.
        # A cover page matching the ticker/year must not displace the numeric row.
        def chunk(identifier, text):
            return Chunk(identifier, identifier, "doc", text, "TEST", 2025,
                         "FY", "10-K", "Statement", "PDF Page 1", "source")
        cover = chunk("cover", "TEST fiscal year 2025 total revenues discussed. " * 12)
        for label in ("Total net sales", "Total revenues", "Net revenues", "Revenue"):
            statement = chunk("statement", f"USD millions\n2025 2024\n{label} 120 100")
            for wording in ("total revenues", "total revenue", "total net sales"):
                hits = bm25(f"What were TEST's {wording} in fiscal year 2025?", [cover, statement])
                self.assertEqual(hits[0].chunk.id, "statement", (label, wording))
                self.assertIn("2025 2024", hits[0].text)
        self.assertEqual(financial_row_label("What was Services revenue?"), "Services")
        self.assertIsNone(financial_row_label("What manufacturing risks were disclosed?"))

    def test_one_correction_keeps_evidence_and_audits_rejected_completion(self):
        evidence={"E1":"Suppliers can\n  delay production."}
        bad=self.payload("Supplier delays are a risk.",["Suppliers can delay production..."])
        good=self.payload("Supplier delays are a risk.",["Suppliers can delay production."])
        def response(payload):
            return SimpleNamespace(id="test",usage=None,choices=[SimpleNamespace(finish_reason="stop",
                message=SimpleNamespace(content=json.dumps(payload)))])
        base={"evidence":evidence,"sources":{"E1":{"citation":"[doc, Page 1]"}}}
        for outputs,expected in [([bad,good],"generated"),([bad,bad],"validation-failed")]:
            with patch("financial_rag.generation.request_groq",side_effect=[response(item) for item in outputs]) as call:
                result=build_generation_chain("test").invoke({"question":"Risks?","evidence":evidence,"base":base})
                self.assertEqual(call.call_count,2)
                self.assertEqual(result["status"],expected)
                repair=json.loads(call.call_args.args[0].to_messages()[1].content)
                self.assertEqual(repair["retrieved_context"],evidence)
                self.assertEqual(len(result["generation_attempts"]),2)
                self.assertIn("rejected_response",result["generation_attempts"][0])


if __name__=="__main__":
    unittest.main()

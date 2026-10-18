"""Context: regress missing dates, audited debt addition and statement selection failures."""

from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from financial_rag.calculator import calculate
from financial_rag.generation import validate_response,finish_answer
from financial_rag.core import Chunk, Index
from financial_rag.retrieval import Hit, retrieve
from financial_rag.retrieval import primary_balance_statement
from financial_rag.statement_tools import prepare_statement_tools


class FinancialValidationRepairTests(unittest.TestCase):
    def test_contents_reference_is_not_a_primary_balance_statement(self):
        self.assertFalse(primary_balance_statement("Index to financial statements Consolidated Balance Sheets as of 2025 and 2024"))
        self.assertTrue(primary_balance_statement("Sample Inc. CONSOLIDATED BALANCE SHEETS\n(In millions)\n2025 2024"))

    def statement_fixture(self):
        # Context: joined company/heading text reproduces the semantic splitter;
        # year columns, scale and component headings remain original source text.
        evidence={"E1":"Sample Inc. CONSOLIDATED BALANCE SHEETS\n(In millions, USD)\n2025 2024\n"
            "Current assets:\nInventories 5,718 7,286\nCurrent liabilities:\nCommercial paper 7,979 9,967\n"
            "Term debt 12,350 10,912\nNon-current liabilities:\nTerm debt 78,328 85,750",
            "E2":"Sample Inc. CONSOLIDATED BALANCE SHEETS\n(In millions, USD)\n2024 2023\n"
            "Current assets:\nInventories 7,286 6,331\nCurrent liabilities:\nCommercial paper 9,967 5,985"}
        sources={key:{"document_id":key,"ticker":"DEMO","fiscal_year":year,"doc_type":"10-K",
            "section_path":"Balance sheet","location":"PDF Page 1","source":"local"}
            for key,year in (("E1",2025),("E2",2024))}
        return evidence,sources

    def test_statement_tools_preserve_thousands_and_cite_both_own_year_filings(self):
        evidence,sources=self.statement_fixture()
        tools=prepare_statement_tools("Using inventories from FY2024 and FY2025 consolidated balance sheets, "
            "give the absolute and percentage change in USD millions; cite both filings.",evidence,sources)
        self.assertEqual([o["value"] for o in tools["observations"]],["5718","7286"])
        self.assertEqual({o["evidence_id"] for o in tools["observations"]},{"E1","E2"})
        self.assertEqual([r["display_output"] for r in tools["calculator_results"]],["-1568.00","-21.52"])
        for observation in tools["observations"]:
            self.assertIn(observation["quote"],evidence[observation["evidence_id"]])
            self.assertIn(observation["period_quote"],evidence[observation["evidence_id"]])

    def test_statement_tools_build_audited_debt_sum_from_separate_components(self):
        evidence,sources=self.statement_fixture()
        tools=prepare_statement_tools("Using the FY2025 consolidated balance sheet, calculate total term debt "
            "by adding current and non-current portions in USD millions.",evidence,sources)
        self.assertEqual(tools["calculator_results"][0]["display_output"],"90678.00")
        self.assertEqual([i["component"] for i in tools["calculator_requests"][0]["inputs"]],
            ["Current liabilities","Non-current liabilities"])

    def test_server_owns_calculation_inputs_and_keeps_model_draft_for_audit(self):
        evidence,sources=self.statement_fixture()
        for key in sources:
            sources[key]["citation"]="["+key+", PDF Page 1]"
        tools=prepare_statement_tools("Using inventories from FY2024 and FY2025 balance sheets, "
            "give the percentage change in USD millions; cite both filings.",evidence,sources)
        draft={"answerable":True,"claims":[],"calculations":[{"operation":"unsafe-model-draft","inputs":[]}],"missing":[]}
        response=SimpleNamespace(id="mock",usage=None,choices=[SimpleNamespace(finish_reason="stop",
            message=SimpleNamespace(content=json.dumps(draft)))])
        result=finish_answer({"provider":"nebius","evidence":evidence,"sources":sources,
            "statement_tools":tools},response,"test")
        self.assertEqual(result["status"],"generated")
        self.assertEqual(result["model_calculation_draft"],draft["calculations"])
        self.assertEqual([c["display_output"] for c in result["calculations"]],["-1568.00","-21.52"])
        draft["claims"]=[{"text":"Inventories fell 21.53%.","sources":[{"evidence_id":"E1","quote":"Inventories 5,718 7,286"}]}]
        response.choices[0].message.content=json.dumps(draft)
        rejected=finish_answer({"provider":"nebius","evidence":evidence,"sources":sources,
            "statement_tools":tools},response,"test")
        self.assertEqual(rejected["status"],"validation-failed")

    def test_value_only_dividend_quote_recovers_actual_dated_sentence(self):
        sentence="As of September 28, 2024, the quarterly cash dividend was $0.25 per share."
        payload={"answerable":True,"claims":[{"text":"The dividend was $0.25 per share as of September 28, 2024.",
            "sources":[{"evidence_id":"E1","quote":"$0.25 per share"}]}]}
        validate_response(payload,{"E1":sentence+" Annual increases are subject to declaration by the Board."})
        source=payload["claims"][0]["sources"][0]
        self.assertEqual(source["quote"],sentence)
        self.assertIn("original sentence",source["context_added"])

    def test_missing_date_cannot_be_borrowed_from_unrelated_sentence(self):
        payload={"answerable":True,"claims":[{"text":"Dividend $0.25 as of September 28, 2024.",
            "sources":[{"evidence_id":"E1","quote":"Dividend $0.25"}]}]}
        with self.assertRaisesRegex(ValueError,"numerals"):
            validate_response(payload,{"E1":"Dividend $0.25. Revenue rose as of September 28, 2024."})

    def debt_inputs(self):
        common={"currency":"USD","scale":"millions","basis":"reported","scope":"consolidated",
            "period":"FY2025","period_type":"year_end","metric":"Term debt","evidence_id":"E1"}
        return [{**common,"value":"12350","component":"Current liabilities",
                 "quote":"Current liabilities:\nTerm debt 12,350 10,912"},
                {**common,"value":"78328","component":"Non-current liabilities",
                 "quote":"Non-current liabilities:\nTerm debt 78,328 85,750"}]

    def test_term_debt_sum_has_exact_decimal_audit(self):
        inputs=self.debt_inputs()
        result=calculate("sum",inputs,{"E1":"\n".join(i["quote"] for i in inputs)})
        self.assertEqual(result["display_output"],"90678.00")
        self.assertEqual(result["formula"],"first component + second component")

    def test_sum_rejects_duplicate_cross_period_and_wrong_component_amounts(self):
        original=self.debt_inputs(); evidence={"E1":"\n".join(i["quote"] for i in original)}
        for change in ({"period":"FY2024"},{"component":"Current liabilities"},{"component":None}):
            operands=deepcopy(original);operands[1].update(change)
            with self.assertRaises(ValueError):
                calculate("sum",operands,evidence)
        operands=deepcopy(original)
        operands[0].update(value="78328",quote=evidence["E1"])
        with self.assertRaisesRegex(ValueError,"within its cited component"):
            calculate("sum",operands,evidence)

    def test_zero_component_is_valid_for_sum(self):
        inputs=self.debt_inputs();inputs[1].update(value="0",quote="Non-current liabilities:\nTerm debt 0")
        self.assertEqual(calculate("sum",inputs,{"E1":"\n".join(i["quote"] for i in inputs)})["display_output"],"12350.00")

    def test_inventory_changes_are_calculator_outputs_not_model_claims(self):
        common={"currency":"USD","scale":"millions","basis":"reported","scope":"consolidated",
                "period_type":"year_end","metric":"inventories","evidence_id":"E1","quote":"Inventories 5,718 7,286"}
        operands=[{**common,"value":"5718","period":"FY2025"},{**common,"value":"7286","period":"FY2024"}]
        evidence={"E1":common["quote"]}
        self.assertEqual(calculate("difference",deepcopy(operands),evidence)["display_output"],"-1568.00")
        self.assertEqual(calculate("growth",deepcopy(operands),evidence)["display_output"],"-21.52")
        payload={"answerable":True,"claims":[{"text":"Inventories fell 21.53%.",
            "sources":[{"evidence_id":"E1","quote":common["quote"]}]}]}
        with self.assertRaisesRegex(ValueError,"numerals"):
            validate_response(payload,evidence)

    def test_requested_statements_survive_rerank_and_preserve_company_scope(self):
        first=Chunk("a","a","d2025","CONSOLIDATED BALANCE SHEETS\n(In millions)\n2025 2024\nCurrent assets:\nInventories 5,718 7,286\nCurrent liabilities:\nCommercial paper 7,979 9,967",
            "DEMO",2025,"FY","10-K","Balance sheet","PDF Page 1","local")
        second=replace(first,id="b",parent_id="b",doc_id="d2024",fiscal_year=2024,
            text="CONSOLIDATED BALANCE SHEETS\n2024 2023\nCurrent assets:\nInventories 7,286 6,331\nCurrent liabilities:\nCommercial paper 9,967 5,985")
        prose=replace(first,id="p",parent_id="p",text="Inventories are discussed here; commercial paper was $8.0 billion.")
        foreign=replace(first,id="x",parent_id="x",ticker="OTHER")
        chunks=[first,second,prose,foreign]
        index=Index({"strategy":"enhanced"},chunks,{c.id:c.text for c in chunks})
        with patch("financial_rag.retrieval.flashrank",side_effect=lambda hits,q:sorted(hits,key=lambda h:h.chunk.id!="p")):
            selected,pool=retrieve(index,"Using inventories from both balance sheets, cite both filings.",
                filters={"ticker":"DEMO","fiscal_year":[2024,2025]},candidates=4,top_n=2,rerank=True)
        self.assertEqual({h.chunk.fiscal_year for h in selected},{2024,2025})
        self.assertTrue(all(h.chunk.id in {"a","b"} for h in selected))
        self.assertNotIn("x",[p["id"] for p in pool])


if __name__=="__main__":
    unittest.main()

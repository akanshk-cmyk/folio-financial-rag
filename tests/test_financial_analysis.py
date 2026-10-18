"""Context: independently known statement inputs verify arithmetic and fail-closed analysis."""

import copy
import unittest

from financial_rag.core import Chunk
from financial_rag.financial_analysis import analyze_financial_statements
from financial_rag.overview import financial_overview


class FinancialAnalysisTests(unittest.TestCase):
    def fixture(self,gross="120",liability="100",capex=True):
        statements=(
            ("income","CONSOLIDATED STATEMENTS OF OPERATIONS",
             f"Total net sales 200 100\nTotal cost of sales 80 40\nGross margin {gross} 60\n"
             "Total operating expenses 40 20\nOperating income 80 40\n"
             "Income before provision for income taxes 80 40\nProvision for income taxes 20 10\nNet income 60 30\n"),
            ("balance","CONSOLIDATED BALANCE SHEETS",
             f"Total current assets 150 90\nTotal current liabilities {liability} 60\n"
             "Total assets 300 200\nTotal liabilities 120 90\nTotal shareholders’ equity 180 110\n"),
            ("cash","CONSOLIDATED STATEMENTS OF CASH FLOWS",
             "Net cash provided by operating activities 90 45\n"+
             ("Payments for acquisition of property, plant and equipment (30) (15)\n" if capex else "")),
        )
        chunks=[Chunk(key,key,"filing",heading+"\n(In millions)\n2025 2024\n"+rows+
                      ("Cash and cash equivalents 75 40\n" if key=="balance" else ""),
                      "AAPL",2025,"FY","10-K",heading,"PDF Page "+str(n),"local-source")
                for n,(key,heading,rows) in enumerate(statements,1)]
        report=financial_overview(chunks,"AAPL",[2025])[2025]
        return report,{chunk.id:chunk.text for chunk in chunks}

    def values(self,analysis,year="2025"):
        return {metric["key"]:metric["periods"][year] for metric in analysis["metrics"]}

    def test_known_ratios_amounts_and_prior_year_have_full_audit(self):
        report,evidence=self.fixture()
        analysis=report["analysis"]
        values=self.values(analysis)
        expected={"gross_margin":"60.00","operating_margin":"40.00","net_margin":"30.00",
                  "current_ratio":"1.50","working_capital":"50.00","liabilities_assets":"40.00",
                  "cash_conversion":"1.50","free_cash_flow":"60.00",
                  "fcf_margin":"30.00","equity_assets":"60.00","liabilities_equity":"0.67",
                  "cash_assets":"25.00","cash_liabilities":"0.62","fcf_earnings":"1.00",
                  "capital_intensity":"15.00","operating_cash_margin":"45.00"}
        self.assertEqual({key:value["display_output"] for key,value in values.items()},expected)
        self.assertEqual(self.values(analysis,"2024")["free_cash_flow"]["display_output"],"30.00")
        self.assertTrue(all(check["status"]=="passed" for check in analysis["reconciliations"]))
        for key,metric in values.items():
            self.assertEqual(len(metric["inputs"]),3 if key in {"fcf_margin","fcf_earnings"} else 2)
            for item in metric["inputs"]:
                self.assertIn(item["quote"],evidence[item["evidence_id"]])

    def test_mismatch_withholds_affected_metrics_without_changing_inputs(self):
        report,_=self.fixture(gross="115")
        self.assertEqual(report["metrics"]["gross_profit"]["value"],"115")
        values=self.values(report["analysis"])
        self.assertEqual(values["gross_margin"]["status"],"unavailable")
        self.assertEqual(values["operating_margin"]["status"],"unavailable")
        self.assertEqual(values["free_cash_flow"]["status"],"verified")
        self.assertTrue(any(check["status"]=="mismatch" for check in report["analysis"]["reconciliations"]))

    def test_missing_inputs_and_nonpositive_denominators_do_not_create_values(self):
        report,_=self.fixture(liability="0",capex=False)
        values=self.values(report["analysis"])
        self.assertEqual(values["current_ratio"]["status"],"unavailable")
        self.assertEqual(values["free_cash_flow"]["status"],"unavailable")
        for key in ("fcf_margin","fcf_earnings","capital_intensity"):
            self.assertEqual(values[key]["status"],"unavailable")
        self.assertEqual(values["working_capital"]["display_output"],"150.00")

    def test_incompatible_units_and_unverified_numbers_are_rejected(self):
        report,evidence=self.fixture()
        changed=copy.deepcopy(report)
        changed["balance_sheet"]["current_assets"]["unit"]="USD thousands"
        values=self.values(analyze_financial_statements(changed,evidence))
        self.assertEqual(values["current_ratio"]["status"],"unavailable")
        changed=copy.deepcopy(report)
        changed["cash_flow"]["operating_cash"]["value"]="999999"
        values=self.values(analyze_financial_statements(changed,evidence))
        self.assertEqual(values["cash_conversion"]["status"],"unavailable")
        self.assertEqual(values["free_cash_flow"]["status"],"unavailable")


if __name__=="__main__":
    unittest.main()

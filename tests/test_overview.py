"""Context: test financial provenance and fiscal-column selection, without APIs."""

import unittest
from unittest.mock import patch

from financial_rag.core import Chunk
from financial_rag.overview import financial_overview, render_statement_table


class OverviewTests(unittest.TestCase):
    def statement(self,year=2025,doc="doc",header="2025 2024",rows=None):
        text="CONSOLIDATED STATEMENTS OF OPERATIONS\n(In millions, except per-share amounts)\n"+header+"\n"+(rows or
            "Total net sales 120 100\nNet income 30 20\nOperating income 40 30\nDiluted $ 1.20 $ 1.00\n")
        return Chunk(doc,doc,doc,text,"AAPL",year,"FY","10-K","Operations","PDF Page 41","source")

    def test_source_values_and_calculator_changes_are_cited(self):
        source=self.statement()
        report=financial_overview([source],"AAPL",[2025])[2025]
        sales=report["metrics"]["sales"]
        self.assertEqual(sales["value"],"120")
        self.assertEqual(sales["prior_value"],"100")
        self.assertEqual(sales["change"]["display_output"],"20.00")
        self.assertIn(sales["quote"],source.text)
        self.assertIn("PDF Page 41",sales["citation"])
        self.assertNotIn("rd",report["metrics"])

    def test_selected_year_company_and_header_must_agree(self):
        self.assertEqual(financial_overview([self.statement()],"MSFT",[2025]),{})
        self.assertEqual(financial_overview([self.statement()],"AAPL",[2024]),{})
        self.assertEqual(financial_overview([self.statement(year=2024)],"AAPL",[2024]),{})

    def test_malformed_numeric_columns_do_not_become_figures(self):
        malformed=self.statement(rows="Total net sales 120 100 80\n")
        self.assertEqual(financial_overview([malformed],"AAPL",[2025]),{})

    def test_conflicting_report_versions_are_not_silently_mixed(self):
        with self.assertRaisesRegex(ValueError,"More than one"):
            financial_overview([self.statement(doc="one"),self.statement(doc="two")],"AAPL",[2025])

    def test_product_categories_use_disaggregated_sales_table(self):
        source=self.statement()
        product=Chunk("product","product","doc",
            "The following table shows disaggregated net sales (in millions):\n2025 2024\niPhone $ 90 $ 80\nServices (1) 30 20\n",
            "AAPL",2025,"FY","10-K","Revenue","PDF Page 49","source")
        report=financial_overview([source,product],"AAPL",[2025])[2025]
        self.assertEqual(report["products"]["iPhone"]["value"],"90")
        self.assertEqual(report["products"]["Services"]["value"],"30")

    def test_share_count_exception_does_not_change_currency_scale(self):
        source=self.statement()
        source.text=source.text.replace("In millions, except per-share amounts",
            "In millions, except number of shares, which are reflected in thousands, and per-share amounts")
        report=financial_overview([source],"AAPL",[2025])[2025]
        self.assertEqual(report["metrics"]["sales"]["unit"],"USD millions")
        self.assertEqual(report["metrics"]["eps"]["unit"],"USD per share")

    def dated_statement(self,header="Jan 25, 2026    Jan 26, 2025    Jan 28, 2024"):
        """Context: reproduce a dated-column income statement without depending on private uploads."""
        return Chunk("n","n","nvda","Consolidated Statements of Income\n"
            "(In millions, except per share data)\nYear Ended\n"+header+"\n"
            "Revenue $ 215,938 $ 130,497 $ 60,922\n"
            "Net income $ 120,067 $ 72,880 $ 29,760\n"
            "Sales, general and administrative 4,579 3,491 2,654\n"
            "Net income per share:\nDiluted $ 4.90 $ 2.94 $ 1.19\n"
            "Weighted average shares used in per share computation:\nDiluted 24,514 24,804 24,940\n",
            "NVDA",2026,"FY","10-K","Income","PDF Page 71","uploaded-pdf:test")

    def test_explicit_date_columns_keep_year_values_and_eps_aligned(self):
        chunk=self.dated_statement()
        report=financial_overview([chunk],"NVDA",[2026],{"nvda":{"currency":"USD"}})[2026]
        self.assertEqual(report["metrics"]["sales"]["value"],"215938")
        self.assertEqual(report["metrics"]["sales"]["prior_value"],"130497")
        self.assertEqual(report["metrics"]["sales"]["unit"],"USD millions")
        self.assertEqual(report["metrics"]["eps"]["value"],"4.90")
        self.assertEqual(report["metrics"]["sga"]["value"],"4579")
        self.assertIn(report["metrics"]["sales"]["quote"],chunk.text)

    def test_full_month_dates_work_and_wrong_year_still_fails(self):
        chunk=self.dated_statement("January 25, 2026    January 26, 2025    January 28, 2024")
        self.assertTrue(financial_overview([chunk],"NVDA",[2026],{"nvda":{"currency":"USD"}}))
        chunk.text=chunk.text.replace("January 25, 2026","January 25, 2027")
        self.assertEqual(financial_overview([chunk],"NVDA",[2026],{"nvda":{"currency":"USD"}}),{})

    def test_share_count_alone_is_not_eps_and_percentages_are_not_currency(self):
        chunk=self.dated_statement()
        chunk.text=chunk.text.replace("Net income per share:\nDiluted $ 4.90 $ 2.94 $ 1.19\n","")
        chunk.text=chunk.text.replace("Revenue $ 215,938 $ 130,497 $ 60,922","Revenue 100.0 % 100.0 % 100.0 %")
        report=financial_overview([chunk],"NVDA",[2026],{"nvda":{"currency":"USD"}})[2026]
        self.assertNotIn("eps",report["metrics"])
        self.assertNotIn("sales",report["metrics"])

    def test_balance_and_cash_flow_keep_statement_scope_signs_and_sources(self):
        source=self.statement()
        balance=Chunk("b","b","doc","CONSOLIDATED BALANCE SHEETS\n(In millions)\n"
            "2025 2024\nCash and cash equivalents 80 60\nTotal assets 500 400\n"
            "Total liabilities 200 150\nTotal shareholders’ equity 300 250\n",
            "AAPL",2025,"FY","10-K","Balance sheets","PDF Page 43","source")
        cash=Chunk("f","f","doc","CONSOLIDATED STATEMENTS OF CASH FLOWS\n(In millions)\n"
            "2025 2024\nNet cash provided by operating activities 100 80\n"
            "Net cash used in investing activities (50) (40)\nDividends paid (9) (8)\n"
            "Cash and cash equivalents at end of period 80 60\n",
            "AAPL",2025,"FY","10-K","Cash flows","PDF Page 46","source")
        foreign=Chunk("foreign","foreign","other-doc",balance.text.replace("500 400","900 800"),
            "AAPL",2025,"FY","10-K","Balance sheets","PDF Page 43","other source")
        report=financial_overview([source,balance,cash,foreign],"AAPL",[2025])[2025]
        assets=report["balance_sheet"]["total_assets"]
        self.assertEqual(assets["value"],"500")
        self.assertEqual(assets["change"]["inputs"][0]["period_type"],"point-in-time")
        self.assertIn(assets["quote"],balance.text)
        self.assertEqual(report["cash_flow"]["operating_cash"]["value"],"100")
        self.assertEqual(report["cash_flow"]["investing_cash"]["value"],"-50")
        self.assertEqual(report["cash_flow"]["dividends"]["value"],"-9")
        self.assertNotIn("change",report["cash_flow"]["dividends"])
        self.assertEqual(report["cash_flow"]["operating_cash"]["evidence_id"],"f")

    def test_absent_statements_remain_unavailable_without_using_income_rows(self):
        report=financial_overview([self.statement()],"AAPL",[2025])[2025]
        self.assertEqual(report["balance_sheet"],{})
        self.assertEqual(report["cash_flow"],{})

    def test_income_statement_follows_financial_order_and_places_margin_under_profit(self):
        source=self.statement(rows="Total net sales 120 100\nTotal cost of sales 50 40\n"
            "Gross margin 70 60\nResearch and development 10 8\n"
            "Selling, general and administrative 5 4\nTotal operating expenses 15 12\n"
            "Operating income 55 48\nOther income/(expense), net (2) 1\n"
            "Income before provision for income taxes 53 49\nProvision for income taxes 13 12\n"
            "Net income 40 37\nEarnings per share:\nBasic 2.00 1.85\nDiluted 1.90 1.80\n")
        metrics=financial_overview([source],"AAPL",[2025])[2025]["metrics"]
        self.assertEqual(metrics["cost_sales"]["value"],"50")
        self.assertEqual(metrics["other_income"]["value"],"-2")
        self.assertEqual(metrics["income_tax"]["value"],"13")
        metrics["gross_margin"]={"title":"Total gross margin","unit":"%","value":"58.3","prior_value":"60.0"}
        with patch("streamlit.markdown") as markdown,patch("streamlit.caption"):
            render_statement_table(dict(reversed(list(metrics.items()))),2025,income_statement=True)
        main_table=markdown.call_args_list[0].args[0]
        titles=["Net sales","Cost of sales / revenue","Gross profit","Total operating expenses",
                "Operating income","Income before income taxes","Income tax expense","Net income","Basic EPS","Diluted EPS"]
        positions=[main_table.index(f'<td>{title}</td>') for title in titles]
        self.assertEqual(positions,sorted(positions))
        self.assertIn("58.30%",main_table)
        self.assertLess(main_table.index('<td>Gross profit</td>'),main_table.index('<td>Gross margin %</td>'))
        self.assertLess(main_table.index('<td>Gross margin %</td>'),main_table.index('Operating expenses</td>'))
        self.assertEqual(markdown.call_count,1)


if __name__=="__main__":
    unittest.main()

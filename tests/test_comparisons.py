"""Context: combined columns must preserve periods, restatements, and source-specific calculations."""

import unittest
from copy import deepcopy
from unittest.mock import patch

from financial_rag.comparisons import combine_reports, year_growth
from financial_rag.core import Chunk
from financial_rag.overview import financial_overview, render_statement_table


class ComparisonTests(unittest.TestCase):
    def reports(self):
        chunks=[]
        for year,sales in ((2025,"120 110"),(2024,"100 90")):
            text=f"CONSOLIDATED STATEMENTS OF OPERATIONS\n(In millions, except per-share amounts)\n{year} {year-1}\n"
            text+=f"Total net sales {sales}\nNet income 30 20\nOperating income 40 30\nDiluted 1.20 1.00\n"
            if year==2024:
                text+="Total cost of sales 60 50\n"
            chunks.append(Chunk(str(year),str(year),str(year),text,"AAPL",year,"FY","10-K","Income","PDF Page 1","source"))
        return financial_overview(chunks,"AAPL",[2024,2025])

    def test_single_and_combined_year_columns_match_selection(self):
        reports=self.reports()
        self.assertEqual(combine_reports(reports,[2025])["comparison_years"],[2025,2024])
        self.assertEqual(combine_reports(reports,[2024])["comparison_years"],[2024,2023])
        merged=combine_reports(reports,[2024,2025])
        self.assertEqual(merged["comparison_years"],[2025,2024,2023])
        self.assertEqual(merged["metrics"]["sales"]["period_values"],{2025:"120",2024:"110",2023:"90"})
        self.assertEqual(merged["metrics"]["cost_sales"]["value"],None)
        self.assertNotIn("change",merged["metrics"]["cost_sales"])
        self.assertEqual(reports[2024]["metrics"]["sales"]["value"],"100")

    def test_newer_filing_controls_overlap_and_complete_ratio_operands(self):
        merged=combine_reports(self.reports(),[2024,2025])
        source=merged["metrics"]["sales"]["period_sources"][2024]
        self.assertEqual(source["document_id"],"2025")
        self.assertTrue(any(item["chosen_value"]=="110" and item["other_value"]=="100" for item in merged["comparison_conflicts"]))
        margin=next(metric for metric in merged["analysis"]["metrics"] if metric["key"]=="net_margin")
        self.assertEqual(margin["periods"]["2024"]["inputs"][1]["value"],"110")
        self.assertEqual(margin["periods"]["2023"]["inputs"][1]["value"],"90")
        self.assertTrue(all(item["document_id"]=="2025" for item in margin["periods"]["2024"]["inputs"]))

    def test_combined_statement_html_has_three_columns_and_missing_cells(self):
        merged=combine_reports(self.reports(),[2025,2024])
        with patch("streamlit.markdown") as markdown,patch("streamlit.caption"):
            render_statement_table(merged["metrics"],2025,True,merged["comparison_years"])
        table=markdown.call_args_list[0].args[0]
        self.assertIn('<th>FY 2025</th><th>FY 2024</th><th>FY 2023</th>',table)
        self.assertIn("YoY FY 2025",table)
        self.assertIn("YoY FY 2024",table)
        self.assertIn("Unavailable",table)

    def test_each_growth_year_uses_its_displayed_cells_and_cited_inputs(self):
        sales=combine_reports(self.reports(),[2024,2025])["metrics"]["sales"]
        self.assertEqual(sales["growth_by_year"][2025]["display_output"],"9.09")
        self.assertEqual(sales["growth_by_year"][2024]["display_output"],"22.22")
        inputs=sales["growth_by_year"][2024]["inputs"]
        self.assertEqual([(item["period"],item["value"]) for item in inputs],[("FY2024","110"),("FY2023","90")])
        changed=deepcopy(sales)
        changed["period_values"][2023]="0"
        self.assertIsNone(year_growth(changed,2024,2025))
        changed["period_values"][2023]="-90"
        self.assertIsNone(year_growth(changed,2024,2025))
        changed=deepcopy(sales)
        changed["period_sources"][2023]["unit"]="USD thousands"
        self.assertIsNone(year_growth(changed,2024,2025))

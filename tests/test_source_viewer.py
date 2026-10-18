"""Context: evidence previews must preserve quotes and never execute document HTML."""

from html import unescape
import re
import unittest

from financial_rag.source_viewer import source_details, excerpt_html, source_card_html


class SourceViewerTests(unittest.TestCase):
    def test_metadata_and_legacy_citation_preserve_section_commas(self):
        source={"citation":"[AAPL_2025_FY_10-K_PDF_hash, Note 2, Revenue, PDF Page 49 (printed page 36)]"}
        details=source_details(source)
        self.assertEqual(details["title"],"FY 2025 annual report")
        self.assertEqual(details["section"],"Note 2, Revenue")
        self.assertEqual(details["location"],"PDF Page 49 (printed page 36)")
        self.assertEqual(source_details({"citation":"[TEST_2025_10-Q_hash, Income, Page 3]"})["title"],"FY 2025 10-Q")

    def test_literal_highlight_escapes_untrusted_markup_and_preserves_table_spacing(self):
        passage="2025         2024\nRevenue      120      100\n<script>alert('x')</script>"
        html,tabular=excerpt_html(passage,["Revenue      120      100"])
        self.assertTrue(tabular)
        self.assertIn("<mark>Revenue      120      100</mark>",html)
        self.assertNotIn("<script>",html)
        self.assertIn("&lt;script&gt;",html)
        self.assertEqual(unescape(re.sub(r"</?(?:mark|div)[^>]*>","",html)),passage)

    def test_missing_quote_is_not_fabricated_and_synthetic_source_is_labeled(self):
        self.assertEqual(excerpt_html("Revenue 120",["Revenue 999"]),("",False))
        html=source_card_html({"synthetic":True,"citation":"[TEST, Results, Page 1]"},"Revenue 120",["Revenue 120"])
        self.assertIn("NOT AN ACTUAL FILING",html)
        self.assertNotIn("218.6",html)

    def test_overlapping_quotes_do_not_duplicate_source_text(self):
        html,_=excerpt_html("Operating cash flow was 120 million.",["cash flow was 120","120 million"])
        self.assertEqual(html.count("<mark>"),1)
        self.assertIn("cash flow was 120 million",html)


if __name__=="__main__":
    unittest.main()

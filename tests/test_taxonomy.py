"""Context: taxonomy retrieval must bridge vocabulary without merging measures."""

import unittest

from financial_rag.core import Chunk
from financial_rag.retrieval import bm25
from financial_rag.taxonomy import BY_KEY, CONCEPTS, has_numeric_row, match_concepts


class TaxonomyTests(unittest.TestCase):
    def keys(self, query):
        return {item.key for item in match_concepts(query)}

    def test_synonyms_and_punctuation(self):
        for query, expected in (("R&D expense", "research_development"),
                                ("SG&A", "selling_general_admin"),
                                ("stockholders’ equity", "equity"),
                                ("cash flow from operations", "operating_cash"),
                                ("Net earnings", "net_income")):
            self.assertIn(expected, self.keys(query))

    def test_embedded_generic_terms_do_not_override_specific_measure(self):
        self.assertEqual(self.keys("cost of revenue"), {"cost_sales"})
        self.assertEqual(self.keys("net profit margin"), {"net_margin"})
        self.assertEqual(self.keys("inventory turnover"), {"inventory_turnover"})
        self.assertEqual(self.keys("adjusted EBITDA"), {"adjusted_ebitda"})
        self.assertEqual(self.keys("adjusted net income"), {"adjusted_net_income"})
        self.assertEqual(self.keys("diluted net income per share"), {"diluted_eps"})
        self.assertEqual(self.keys("gross margin"), {"gross_margin_ambiguous"})

    def test_multi_metric_question_retains_disjoint_concepts(self):
        self.assertEqual(self.keys("Compare net income and operating cash flow YoY"),
                         {"net_income", "operating_cash", "yoy"})
        self.assertNotIn("operating_cash", self.keys("office workflow"))

    def test_kind_and_source_label_boundaries(self):
        self.assertTrue(has_numeric_row("Total net sales $ 120 20% $ 100", BY_KEY["revenue"]))
        self.assertFalse(has_numeric_row("Gross margin 46.9%", BY_KEY["gross_profit"]))
        self.assertTrue(has_numeric_row("Gross margin 46.9%", BY_KEY["gross_margin"]))
        self.assertFalse(has_numeric_row("Diluted 120 100", BY_KEY["diluted_eps"]))
        self.assertFalse(has_numeric_row("Net income 120 100", BY_KEY["adjusted_net_income"]))
        self.assertFalse(has_numeric_row("Operating income 120 100", BY_KEY["ebitda"]))
        self.assertFalse(has_numeric_row("Property and equipment, net 120 100", BY_KEY["capex"]))

    def test_cross_vocabulary_ranking(self):
        def chunk(key, text):
            return Chunk(key, key, "doc", text, "TEST", 2025, "FY", "10-K", "Statement", "Page 1", "source")
        statement = chunk("statement", "2025 2024\nNet cash provided by operating activities 120 100")
        prose = chunk("prose", "Operating cash flow is discussed. " * 20)
        self.assertEqual(bm25("What was operating cash flow?", [prose, statement])[0].chunk.id, "statement")

    def test_curated_catalog_has_unique_ids_and_explicit_cautions(self):
        self.assertEqual(len(BY_KEY), len(CONCEPTS))
        for item in CONCEPTS:
            self.assertTrue(item.aliases)
            self.assertTrue(item.caution)


if __name__ == "__main__":
    unittest.main()

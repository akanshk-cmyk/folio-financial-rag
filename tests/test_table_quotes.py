"""Context: preserve complete table quotations while rejecting spliced column quotes."""

import unittest

from financial_rag.generation import validate_response
from streamlit.testing.v1 import AppTest


class MultiYearTableQuoteTests(unittest.TestCase):
    def test_complete_row_and_year_header_support_all_three_years(self):
        # Context: this reproduces the retrieved iPhone table's column order.
        header="2024             2023             2022"
        row="iPhone  $ 201,183  $ 200,583  $ 205,489"
        evidence={"E1":"Net sales (in millions):\n"+header+"\n"+row}
        payload={"answerable":True,"claims":[{"text":f"iPhone net sales in {year} were ${value} million.",
            "sources":[{"evidence_id":"E1","quote":header},{"evidence_id":"E1","quote":row}]}
            for year,value in ((2022,"205,489"),(2023,"200,583"),(2024,"201,183"))],
            "calculations":[],"missing":[]}
        self.assertEqual(validate_response(payload,evidence),[])

    def test_splicing_label_to_later_column_remains_rejected(self):
        evidence={"E1":"2024 2023 2022\niPhone $ 201,183 $ 200,583 $ 205,489"}
        payload={"answerable":True,"claims":[{"text":"iPhone net sales in 2022 were $205,489 million.",
            "sources":[{"evidence_id":"E1","quote":"iPhone $ 205,489"}]}],"calculations":[],"missing":[]}
        with self.assertRaisesRegex(ValueError,"invalid source quote"):
            validate_response(payload,evidence)

    def test_ui_does_not_describe_validation_failure_as_missing_financial_data(self):
        # Context: even historical failed results should display an accurate explanation.
        app=AppTest.from_string('''from streamlit_app import show_result
show_result({"status":"validation-failed", "answer":"Information not available in provided financial disclosures.",
    "validation":"Claim contains an invalid source quote.","question":"iPhone sales?"})''').run()
        self.assertFalse(list(app.exception))
        self.assertTrue(any("does not mean the figures are missing" in item.value for item in app.warning))
        self.assertFalse(any(item.value=="Information not available in provided financial disclosures." for item in app.markdown))


if __name__=="__main__":
    unittest.main()

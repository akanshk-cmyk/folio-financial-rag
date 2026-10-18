"""Context: exercise the actual UI without starting models or hosted requests."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from financial_rag.core import Chunk, Index
from financial_rag.demo_ui import ROOT, available_indexes, index_uploaded_pdf, run_local_question, supporting_evidence


class StreamlitDemoTests(unittest.TestCase):
    """Context: validate source filters and explicit execution in the local demo."""

    def test_single_submit_generates_answer_with_cited_evidence(self):
        text = ("CONSOLIDATED STATEMENTS OF OPERATIONS\n(In millions, except per-share amounts)\n"
                "2025 2024\nTotal net sales 120 100\nNet income 30 20\nOperating income 40 30\nDiluted $ 1.20 $ 1.00\n")
        chunk = Chunk("c", "p", "AAPL_2025", text, "AAPL", 2025,
                      "FY", "10-K", "Results", "PDF Page 1", "https://example.com/filing")
        product = Chunk("products","products","AAPL_2025",
            "Disaggregated net sales (in millions):\n2025 2024\niPhone 90 80\nServices 30 20\n",
            "AAPL",2025,"FY","10-K","Revenue","PDF Page 49","https://example.com/filing")
        index = Index({"strategy": "enhanced", "backend": "lexical"}, [chunk,product], {"p": chunk.text,"products":product.text})
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder)/"index.json"
            index.save(path)
            with patch("financial_rag.demo_ui.available_indexes", return_value=[path]):
                with patch("financial_rag.demo_ui.save_answer", return_value=ROOT/"runs/test-ui-answer.json"):
                    with patch("financial_rag.demo_ui.run_local_question", wraps=run_local_question) as run:
                        app = AppTest.from_file(str(ROOT/"streamlit_app.py"), default_timeout=15).run()
                        self.assertEqual(list(app.exception), [])
                        run.assert_not_called()
                        navigation=next(radio for radio in app.radio if radio.label=="Workspace navigation")
                        self.assertEqual(navigation.value,"Company overview")
                        self.assertTrue({"Profitability","Financial position","Cash quality"}
                                        <= {tab.label for tab in app.tabs})
                        self.assertEqual(next(metric.value for metric in app.metric if metric.label=="Net sales"),"120")
                        self.assertEqual(next(metric.value for metric in app.metric if metric.label=="Diluted EPS"),"1.20")
                        navigation.set_value("Ask a question").run()
                        app.text_area[0].set_value("What were total net sales?")
                        self.assertFalse(any(button.label=="Find evidence" for button in app.button))
                        self.assertTrue(any(button.label=="Answer" for button in app.button))
                        next(radio for radio in app.radio if radio.label=="Generation provider").set_value("Nebius").run()
                        with patch.dict("os.environ",{"NEBIUS_API_KEY":""}):
                            next(button for button in app.button if button.label=="Answer").click().run()
                        self.assertEqual(run.call_count,0)
                        self.assertTrue(any("NEBIUS_API_KEY" in error.value for error in app.error))
                        # Context: verify provider routing without making a paid generation request.
                        run.return_value={"question":"What were total net sales?", "status":"generated",
                            "workflow_trace":[{"stage":"router","topics":["revenue"]}],
                            "answer":"Mock grounded answer.", "claims":[{"sources":[{"evidence_id":"E1", "quote":"Total net sales 120 100"}]}],
                            "evidence":{"E1":text,"E2":"Unrelated passage"},
                            "sources":{"E1":{"citation":"[AAPL_2025, PDF Page 1]","source":"https://example.com/filing"}}}
                        with patch.dict("os.environ",{"NEBIUS_API_KEY":"test-only"}):
                            next(button for button in app.button if button.label=="Answer").click().run()
                        self.assertEqual(list(app.exception),[])
                        self.assertEqual(run.call_count,1)
                        self.assertEqual(run.call_args.args[4],"nebius")
                        self.assertTrue(run.call_args.kwargs["adaptive"])
                        self.assertTrue(any(expander.label=="Answer workflow" for expander in app.expander))
                        self.assertTrue(any(heading.value=="Supporting evidence" for heading in app.subheader))
                        self.assertEqual([element.value for element in app.text], ["Total net sales 120 100"])
                        # Context: opening a source uses saved evidence without another answer request.
                        next(button for button in app.button if button.label=="View source").click().run()
                        self.assertEqual(list(app.exception),[])
                        self.assertEqual(run.call_count,1)
                        self.assertTrue(any("source-viewer" in block.value for block in app.markdown))

    def test_supporting_evidence_includes_calculations_and_rejects_uncited_text(self):
        # Context: display only exact cited excerpts, deduplicated by saved evidence ID.
        source={"evidence_id":"E1","quote":"Revenue 120"}
        result={"claims":[{"sources":[source,source,{"evidence_id":"E1","quote":"Invented 999"}]}],
                "calculations":[{"inputs":[{"evidence_id":"E2","quote":"Revenue 100"}]}],
                "evidence":{"E1":"Revenue 120\nOther 30","E2":"Revenue 100","E3":"Unrelated"}, "sources":{}}
        entries=supporting_evidence(result)
        self.assertEqual([entry["identifier"] for entry in entries],["E1","E2"])
        self.assertEqual(entries[0]["quotes"],["Revenue 120"])

    def test_uploaded_pdf_rejects_non_pdf_before_parsing(self):
        with self.assertRaisesRegex(ValueError, "PDF"):
            index_uploaded_pdf(b"not a pdf", "AAPL", 2025)

    def test_local_question_filters_year_and_company(self):
        chunks = [Chunk(str(i), str(i), str(i), f"Revenue {value} million", ticker, year,
                        "FY", "10-K", "Results", "PDF Page 1", "source")
                  for i, (ticker,year,value) in enumerate([("AAPL",2024,100), ("AAPL",2025,120), ("MSFT",2025,200)])]
        index = Index({}, chunks, {chunk.id: chunk.text for chunk in chunks})
        result = run_local_question(index, "Revenue", "AAPL", [2025])
        self.assertEqual(result["status"], "preview-only")
        self.assertEqual(len(result["evidence"]), 1)
        self.assertIn("120", next(iter(result["evidence"].values())))


if __name__ == "__main__":
    unittest.main()

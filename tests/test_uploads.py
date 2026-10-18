"""Context: exercise persistent company uploads with real PDF parsing, without hosted services."""

import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pymupdf

from financial_rag.core import Index, load_documents
from financial_rag.demo_ui import available_indexes, index_uploaded_pdf, run_local_question
from financial_rag.overview import financial_overview


class FakeEmbedder:
    """Context: deterministic vectors check alignment and persistence, not model quality."""
    name="test-encoder"
    revision="test-revision"

    def encode(self,texts,query=False):
        return [[1.0,0.0] for _ in texts]


class UploadTests(unittest.TestCase):
    def pdf(self,year=2025):
        # Context: layout includes explicit years and scale, just as the extractor requires.
        with pymupdf.open() as document:
            page=document.new_page()
            page.insert_text((40,40),f"CONSOLIDATED STATEMENTS OF INCOME\n"
                f"(USD in thousands, except per-share amounts)\n{year} {year-1}\n"
                "Total revenues 1,200 1,000\nNet earnings 300 200\n"
                "Income from operations 400 300\nDiluted earnings per share 1.20 1.00\n")
            return document.tobytes()

    def test_persist_reload_search_and_company_overview(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            index=index_uploaded_pdf(self.pdf(),"MSFT",2025,company_name="Example Corporation",
                                     storage_dir=root/"data/uploads")
            saved=Path(index.manifest["saved_index"])
            self.assertTrue(saved.exists())
            self.assertTrue((saved.parent/"source.pdf").exists())
            documents=load_documents(saved.parent/"document.json")
            self.assertEqual(Path(documents[0].provenance["local_artifact"]),saved.parent/"source.pdf")
            reloaded=Index.load(saved)
            with patch("financial_rag.demo_ui.ROOT",root):
                self.assertIn(saved,available_indexes())
            report=financial_overview(reloaded.chunks,"MSFT",[2025],reloaded.manifest["upload_documents"])[2025]
            self.assertEqual(report["company_name"],"Example Corporation")
            self.assertEqual(report["metrics"]["sales"]["unit"],"USD thousands")
            self.assertEqual(report["metrics"]["sales"]["change"]["display_output"],"20.00")
            self.assertEqual(report["metrics"]["eps"]["unit"],"USD per share")
            self.assertEqual(report["products"],{})
            result=run_local_question(reloaded,"What were total revenues?","MSFT",[2025])
            self.assertEqual(result["status"],"preview-only")
            self.assertTrue(result["evidence"])

    def test_dense_merge_retains_vector_alignment_and_metadata(self):
        encoder=FakeEmbedder()
        first=index_uploaded_pdf(self.pdf(2024),"MSFT",2024,embedder=encoder)
        content=self.pdf()
        second=index_uploaded_pdf(content,"MSFT",2025,current=first,embedder=encoder)
        self.assertEqual(len(second.vectors),len(second.chunks))
        self.assertEqual(second.manifest["documents"],2)
        self.assertEqual(len(second.manifest["upload_documents"]),2)
        duplicate=index_uploaded_pdf(content,"MSFT",2025,current=second,embedder=encoder)
        # Context: identical uploaded bytes must not add duplicate documents or vectors.
        self.assertEqual(duplicate.manifest["documents"],2)
        self.assertEqual(len(duplicate.vectors),len(second.vectors))
        with self.assertRaisesRegex(ValueError,"settings must match"):
            index_uploaded_pdf(self.pdf(),"MSFT",2025,current=first)

    def test_invalid_upload_creates_no_saved_index(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for content in (b"not a pdf",b"%PDF-broken"):
                with self.assertRaises(Exception):
                    index_uploaded_pdf(content,"MSFT",2025,storage_dir=root)
            self.assertEqual(list(root.rglob("index.json")),[])
            with self.assertRaisesRegex(ValueError,"fiscal year"):
                index_uploaded_pdf(self.pdf(),"MSFT",1700,storage_dir=root)

    def test_upload_form_activates_persisted_company_report(self):
        """Context: a successful UI submission must reload its saved index and company overview."""
        from streamlit.testing.v1 import AppTest
        from financial_rag.demo_ui import ROOT
        content=self.pdf()
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            with patch("financial_rag.demo_ui.ROOT",Path(folder)):
                with patch("streamlit.file_uploader",return_value=SimpleNamespace(getvalue=lambda:content)):
                    app=AppTest.from_file(str(ROOT/"streamlit_app.py"),default_timeout=20).run()
                    next(radio for radio in app.radio if radio.label=="Workspace navigation").set_value("Document library").run()
                    next(field for field in app.text_input if field.label=="Company name").set_value("Example Corporation")
                    next(field for field in app.text_input if field.label=="Filing ticker").set_value("MSFT")
                    next(button for button in app.button if button.label=="Save and index filing").click().run()
                    self.assertEqual(list(app.exception),[])
                    self.assertIn("Example Corporation",[title.value for title in app.title])
                    self.assertEqual(next(metric.value for metric in app.metric if metric.label=="Net sales"),"1,200")
                    selected=next(field.value for field in app.selectbox if field.label=="Filing index")
                    self.assertTrue((Path(folder)/selected).is_file())

    def test_conflicting_currency_is_not_relabelled(self):
        index=index_uploaded_pdf(self.pdf(),"MSFT",2025,currency="EUR")
        self.assertEqual(financial_overview(index.chunks,"MSFT",[2025],index.manifest["upload_documents"]),{})


if __name__=="__main__":
    unittest.main()

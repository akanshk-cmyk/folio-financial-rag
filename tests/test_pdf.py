"""Context: optional PDF tests protect page citations and financial source integrity.

PDF fixtures are generated locally; no real financial report or network is needed.
The non-PDF tests still run when PDF extras are not installed.
"""

import importlib.util
from pathlib import Path
import tempfile
import unittest

from financial_rag.chunking import build_index
from financial_rag.ingest import parse_document
from financial_rag.pdf import clean_page_text


@unittest.skipUnless(importlib.util.find_spec("pymupdf"), "Install the pdf extra to run PDF tests")
class PDFTests(unittest.TestCase):
    """Context: retain physical pages even when printed numbering differs."""

    def setUp(self):
        import pymupdf
        self.pdf = pymupdf
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "fixture.pdf"

    def tearDown(self):
        self.temporary.cleanup()

    def write_fixture(self):
        with self.pdf.open() as doc:
            first = doc.new_page()
            first.insert_text((40, 40), "CONSOLIDATED STATEMENTS OF OPERATIONS\nYears ended 2025 2024\nRevenue 120 100\nOperating income (20) 15\nApple Inc. | 2025 Form 10-K | 29")
            second = doc.new_page()
            second.insert_text((40, 40), "Notes to financial statements\nRevenue includes services.\nApple Inc. | 2025 Form 10-K | 30")
            doc.new_page()  # Context: intentionally blank page must be recorded.
            doc.save(self.path)

    def test_pages_values_and_provenance(self):
        self.write_fixture()
        doc = parse_document(self.path, "DEMO", 2025)
        self.assertEqual(doc.sections[0].location, "PDF Page 1 (printed page 29)")
        self.assertEqual(doc.sections[1].location, "PDF Page 2 (printed page 30)")
        self.assertIn("(20)", doc.sections[0].text)
        self.assertEqual(doc.provenance["blank_pages"], [3])
        self.assertEqual(doc.provenance["pages"], 3)
        self.assertEqual(len(doc.content_hash), 64)

    def test_pdf_parents_never_merge_different_page_locations(self):
        self.write_fixture()
        doc = parse_document(self.path, "DEMO", 2025)
        index = build_index([doc], "enhanced")
        self.assertEqual(len(index.chunks), 2)
        self.assertNotEqual(index.chunks[0].parent_id, index.chunks[1].parent_id)
        self.assertNotIn("includes services", index.parents[index.chunks[0].parent_id])
        self.assertIn("PDF Page 1", index.chunks[0].citation())

    def test_image_only_page_requires_ocr(self):
        with self.pdf.open() as image_source:
            image = image_source.new_page().get_pixmap().tobytes("png")
        with self.pdf.open() as doc:
            page = doc.new_page()
            page.insert_image(page.rect, stream=image)
            doc.save(self.path)
        with self.assertRaisesRegex(ValueError, "OCR"):
            parse_document(self.path, "DEMO", 2025)

    def test_encrypted_pdf_is_rejected(self):
        with self.pdf.open() as doc:
            doc.new_page().insert_text((40, 40), "Confidential test fixture")
            doc.save(self.path, encryption=self.pdf.PDF_ENCRYPT_AES_256,
                     owner_pw="owner", user_pw="user")
        with self.assertRaisesRegex(ValueError, "Password-protected"):
            parse_document(self.path, "DEMO", 2025)

    def test_no_guessed_printed_page(self):
        with self.pdf.open() as doc:
            doc.new_page().insert_text((40, 40), "Revenue 120")
            doc.save(self.path)
        result = parse_document(self.path, "DEMO", 2025)
        self.assertEqual(result.sections[0].location, "PDF Page 1")


class CleaningTests(unittest.TestCase):
    """Context: browser furniture can be removed without collapsing numeric columns."""

    def test_spacing_signs_and_footnotes_remain(self):
        raw = ("10/16/26, 1:35 PM     report\nRevenue     120     (100)\n"
               "Footnote 1: retained\nhttps://www.sec.gov/Archives/report.htm     1/78")
        text = clean_page_text(raw)
        self.assertIn("Revenue     120     (100)", text)
        self.assertIn("Footnote 1", text)
        self.assertNotIn("1:35 PM", text)
        self.assertNotIn("https://", text)


if __name__ == "__main__":
    unittest.main()

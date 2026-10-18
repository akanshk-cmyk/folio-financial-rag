"""Context: verify persisted document isolation and grounded QA without API/model downloads."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from financial_rag.core import Chunk, Document, Index, Section
from financial_rag.document_qa import DocumentQA, DocumemtQA


class TestEmbedder:
    name, revision = "test-encoder", "pinned-revision"

    def encode(self, texts, query=False):
        return [[1.0, 0.0] for _ in texts]


class DocumentQATests(unittest.TestCase):
    def snapshot(self, folder):
        chunks = [Chunk(str(n), str(n), identity, "USD millions\n2025 2024\nRevenue 120 100",
                        ticker, 2025, "FY", "10-K", "Income statement", "PDF Page 3", url)
                  for n, (identity, ticker, url) in enumerate((
                      ("A", "AAA", "https://issuer.test/a.pdf"),
                      ("B", "BBB", "https://issuer.test/b.pdf")))]
        index = Index({"strategy":"enhanced", "embedding_model":TestEmbedder.name,
                       "embedding_revision":TestEmbedder.revision}, chunks,
                      {chunk.parent_id:chunk.text for chunk in chunks}, [[1.0,0.0],[1.0,0.0]])
        path = Path(folder)/"index.json"
        index.save(path)
        return path

    def test_load_vectors_search_and_generate_with_only_selected_source(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.snapshot(folder)
            qa = DocumentQA(path, embedder=TestEmbedder(), provider="openai")
            qa.load_document("https://issuer.test/a.pdf")
            hits = qa.current_knowledge_base.search("What was revenue?")
            self.assertEqual({hit.chunk.doc_id for hit in hits}, {"A"})
            payload={"answerable":True,"claims":[{"text":"FY2025 revenue was USD 120 million.",
                      "sources":[{"evidence_id":"E1","quote":hits[0].text}]}],"calculations":[],"missing":[]}
            response=SimpleNamespace(id="test", usage=None, choices=[SimpleNamespace(finish_reason="stop",
                         message=SimpleNamespace(content=json.dumps(payload)))])
            with patch.dict("os.environ", {"OPENAI_API_KEY":"test-only"}), patch(
                    "financial_rag.generation.request_openai", return_value=response) as generate:
                result=qa.ask("What was revenue in FY2025?")
            generate.assert_called_once()
            self.assertEqual(result["status"],"generated")
            self.assertIn("120",result["answer"])
            self.assertEqual(len(result["supporting_evidence"]),1)
            self.assertEqual({source["source"] for source in result["sources"].values()}, {"https://issuer.test/a.pdf"})

    def test_not_loaded_and_failed_switch_do_not_answer_from_previous_document(self):
        with tempfile.TemporaryDirectory() as folder:
            qa=DocumentQA(self.snapshot(folder),embedder=TestEmbedder(),provider="extractive")
            self.assertIs(DocumemtQA,DocumentQA)
            with self.assertRaisesRegex(RuntimeError,"load a document"):
                qa.ask("Revenue?")
            qa.load_document("https://issuer.test/a.pdf")
            with self.assertRaises(FileNotFoundError):
                qa.load_document("https://issuer.test/missing.pdf")
            with self.assertRaises(RuntimeError):
                qa.ask("Revenue?")

    def test_invalid_or_incompatible_vectors_fail_before_qa(self):
        with tempfile.TemporaryDirectory() as folder:
            path=self.snapshot(folder)
            qa=DocumentQA(path,embedder=TestEmbedder(),provider="extractive")
            for mutation, expected in ((lambda data:data["manifest"].update(embedding_revision="other"),"revision"),
                                       (lambda data:data.update(vectors=[]),"vectors"),
                                       (lambda data:data["vectors"].__setitem__(0,[float('nan'),0]),"invalid")):
                self.snapshot(folder)
                data=json.loads(path.read_text());mutation(data);path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError,expected):
                    qa.load_document("https://issuer.test/a.pdf")

    def test_index_downloaded_pdf_and_reload_without_reembedding(self):
        with tempfile.TemporaryDirectory() as folder:
            pdf=Path(folder)/"source.pdf";pdf.write_bytes(b"%PDF-test")
            store=Path(folder)/"vectors"
            document=Document("A","AAA",2025,"FY","10-K","https://issuer.test/a.pdf","hash",
                              [Section("s","Revenue 120","Income statement","PDF Page 1")])
            qa=DocumentQA(store,embedder=TestEmbedder(),provider="extractive")
            with patch("financial_rag.document_qa.parse_document",return_value=document):
                qa.index_pdf(pdf,source_url=document.source,ticker="AAA",fiscal_year=2025)
            self.assertEqual(len(list(store.glob('*.json'))),1)
            fresh=DocumentQA(store,embedder=TestEmbedder(),provider="extractive")
            with patch.object(fresh.embedder,"encode",side_effect=AssertionError("do not re-embed on load")):
                fresh.load_document(document.source)
            self.assertEqual(fresh.ask("Revenue?")["status"],"preview-only")


if __name__ == "__main__":
    unittest.main()

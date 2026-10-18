"""Context: upload checks use fake Pinecone clients and never modify an account."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from types import SimpleNamespace
import unittest
import json
from unittest.mock import Mock, patch

from financial_rag.cloud import prepare_records, publish, PineconeSearch
from financial_rag.core import Chunk, Index, stable_id


class PineconeUploadTests(unittest.TestCase):
    """Context: reject invalid uploads before writes and verify text survives storage."""

    def index(self):
        chunk = Chunk("a", "p", "doc", "Revenue 120", "DEMO", 2025, "FY",
                      "Synthetic", "Results", "PDF Page 1", "synthetic", True)
        return Index({"strategy": "fixed"}, [chunk], {"p": chunk.text}, [[1.0, 0.0]])

    def client(self, dimension=2, existing=True):
        client = Mock()
        client.list_indexes.return_value.names.return_value = ["financial-rag"] if existing else []
        client.describe_index.return_value = SimpleNamespace(dimension=dimension, metric="cosine",
                                                              status={"ready": True}, host="test")
        target = client.Index.return_value
        target.upsert.return_value = SimpleNamespace(upserted_count=1)
        return client, target

    def test_no_vectors_is_rejected_before_network(self):
        index = self.index()
        index.vectors = []
        with patch("financial_rag.cloud.client_from_environment") as connect:
            with self.assertRaisesRegex(ValueError, "one real dense vector"):
                publish(index, "financial-rag")
            connect.assert_not_called()

    def test_bad_dimensions_and_nan_are_rejected(self):
        index = self.index()
        index.chunks.append(replace(index.chunks[0], id="b"))
        for vector in ([1.0], [float("nan"), 0.0]):
            index.vectors = [[1.0, 0.0], vector]
            with self.assertRaisesRegex(ValueError, "dimensions"):
                prepare_records(index)

    def test_oversized_text_is_rejected_before_network(self):
        index = self.index()
        index.chunks[0] = replace(index.chunks[0], text="x" * 39000)
        with patch("financial_rag.cloud.client_from_environment") as connect:
            with self.assertRaisesRegex(ValueError, "Metadata too large"):
                publish(index, "financial-rag")
            connect.assert_not_called()

    def test_incompatible_index_is_not_modified(self):
        client, target = self.client(dimension=768)
        with patch("financial_rag.cloud.client_from_environment", return_value=client):
            with self.assertRaisesRegex(ValueError, "Existing index was not changed"):
                publish(self.index(), "financial-rag", "fixed")
        client.create_index.assert_not_called()
        target.upsert.assert_not_called()

    def test_explicit_creation_is_required(self):
        client, target = self.client(existing=False)
        with patch("financial_rag.cloud.client_from_environment", return_value=client):
            with self.assertRaisesRegex(ValueError, "--create"):
                publish(self.index(), "financial-rag")
        client.create_index.assert_not_called()

    def test_upload_uses_namespace_and_returns_verified_receipt(self):
        index = self.index()
        client, target = self.client()
        target.fetch.return_value = SimpleNamespace(vectors={"a": SimpleNamespace(
            metadata={"text": "Revenue 120"}, values=[1.0, 0.0])})
        target.describe_index_stats.return_value = SimpleNamespace(namespaces={"fixed": SimpleNamespace(vector_count=1)})
        target.query.return_value = SimpleNamespace(matches=[SimpleNamespace(metadata={"text": "Revenue 120"})])
        with patch("financial_rag.cloud.client_from_environment", return_value=client), redirect_stdout(StringIO()):
            receipt = publish(index, "financial-rag", "fixed")
        self.assertEqual(receipt["verification"], "passed")
        self.assertEqual(receipt["uploaded"], 1)
        self.assertEqual(target.upsert.call_args.kwargs["namespace"], "fixed")
        self.assertEqual(target.upsert.call_args.kwargs["vectors"][0]["metadata"]["text"], "Revenue 120")
        self.assertIn("dataset_id", target.query.call_args.kwargs["filter"])
        target.delete.assert_not_called()

    def test_incomplete_acknowledgment_is_reported(self):
        client, target = self.client()
        target.upsert.return_value = SimpleNamespace(upserted_count=0)
        with patch("financial_rag.cloud.client_from_environment", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "Incomplete batch"):
                publish(self.index(), "financial-rag", "fixed")

    def remote_fixture(self):
        index = self.index()
        index.manifest.update(embedding_model="test", embedding_revision="revision")
        encoder = SimpleNamespace(name="test", revision="revision", encode=Mock(return_value=[[1.0, 0.0]]))
        client, target = self.client()
        target.query.return_value = SimpleNamespace(matches=[SimpleNamespace(
            id="a", score=.9, metadata={"text": "Revenue 120"})])
        return index, encoder, client, target

    def test_remote_search_filters_before_query_and_uses_stored_text(self):
        index, encoder, client, target = self.remote_fixture()
        with patch("financial_rag.cloud.client_from_environment", return_value=client):
            hits, pool = PineconeSearch("financial-rag", "fixed")(
                index, "Revenue?", encoder, {"ticker": "DEMO", "fiscal_year": [2025]})
        self.assertEqual(hits[0].text, "Revenue 120")
        self.assertEqual(pool, [{"id": "a", "score": .9}])
        settings = target.query.call_args.kwargs
        self.assertEqual(settings["namespace"], "fixed")
        self.assertIn({"fiscal_year": {"$in": [2025]}}, settings["filter"]["$and"])
        self.assertIn("dataset_id", settings["filter"]["$and"][0])
        encoder.encode.assert_called_once_with(["Revenue?"], query=True)

    def test_remote_search_rejects_stale_text(self):
        index, encoder, client, target = self.remote_fixture()
        target.query.return_value.matches[0].metadata["text"] = "Different disclosure"
        with patch("financial_rag.cloud.client_from_environment", return_value=client):
            with self.assertRaisesRegex(ValueError, "republish"):
                PineconeSearch("financial-rag", "fixed")(index, "Revenue?", encoder)

    def test_remote_search_replays_candidates_without_second_request(self):
        index, encoder, client, target = self.remote_fixture()
        search = PineconeSearch("financial-rag", "semantic")
        with patch("financial_rag.cloud.client_from_environment", return_value=client):
            _, pool = search(index, "Revenue?", encoder)
            _, repeated = search(index, "Revenue?", encoder, saved_candidates=pool)
        self.assertEqual(pool, repeated)
        self.assertEqual(target.query.call_count, 1)

    def test_remote_search_rejects_wrong_encoder_before_request(self):
        index, encoder, client, target = self.remote_fixture()
        encoder.revision = "wrong"
        with patch("financial_rag.cloud.client_from_environment", return_value=client):
            with self.assertRaisesRegex(ValueError, "model/revision"):
                PineconeSearch("financial-rag", "fixed")(index, "Revenue?", encoder)
        target.query.assert_not_called()

    def test_statement_anchor_is_fetched_and_wrong_dataset_is_rejected(self):
        index,encoder,client,target=self.remote_fixture()
        text="CONSOLIDATED BALANCE SHEETS\n(In millions)\n2025 2024\nCurrent assets:\nInventories 12 10\nCurrent liabilities:\nCommercial paper 7 9"
        index.chunks[0]=replace(index.chunks[0],text=text)
        target.query.return_value=SimpleNamespace(matches=[])
        record=SimpleNamespace(metadata={"text":text,"dataset_id":stable_id(json.dumps(index.manifest,sort_keys=True))})
        target.fetch.return_value=SimpleNamespace(vectors={"a":record})
        with patch("financial_rag.cloud.client_from_environment",return_value=client):
            hits,pool=PineconeSearch("financial-rag","fixed")(index,"Commercial paper from the balance sheet?",encoder,{"ticker":"DEMO"})
            self.assertEqual(hits[0].text,text)
            self.assertEqual(pool[0]["id"],"a")
            target.fetch.assert_called_once_with(ids=["a"],namespace="fixed")
            record.metadata["dataset_id"]="foreign"
            with self.assertRaisesRegex(ValueError,"different dataset"):
                PineconeSearch("financial-rag","fixed")(index,"Commercial paper from the balance sheet?",encoder,{"ticker":"DEMO"})


if __name__ == "__main__":
    unittest.main()

"""Context: prevent asymmetric exclusion or incomplete runs in the judged comparison."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class BenchmarkSubsetTests(unittest.TestCase):
    def prepare(self, rows):
        """Context: run the real artifact preparation helper in a temporary workspace."""
        script=Path(__file__).resolve().parents[1]/"examples/prepare_ragas_subset.py"
        spec=importlib.util.spec_from_file_location("benchmark_subset",script)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            target=root/"runs/apple-pinecone-benchmark-v1"
            target.mkdir(parents=True)
            (target/"questions.json").write_text(json.dumps([{"id":"Q1"},{"id":"Q2"}]))
            (target/"experiment.json").write_text(json.dumps({"manifest":{"provider":"nebius"},"rows":rows}))
            with patch.object(module,"__file__",str(root/"examples/prepare_ragas_subset.py")), redirect_stdout(io.StringIO()):
                module.main()
            return json.loads((target/"ragas-eligible-run.json").read_text())

    def test_one_failure_excludes_question_from_all_configurations(self):
        rows=[{"question_id":q,"configuration":c,"status":"generated","claims":[]}
              for q in ("Q1","Q2") for c in ("A","B","C")]
        rows[1]["status"]="validation-failed"
        result=self.prepare(rows)
        self.assertEqual([r["question_id"] for r in result["rows"]],["Q2"]*3)
        coverage=result["manifest"]["evaluation_coverage"]
        self.assertEqual(coverage["excluded_question_ids"],["Q1"])
        self.assertEqual(coverage["generation_statuses"]["validation-failed"],1)
        self.assertEqual(coverage["response_coverage"]["A"]["generated_without_claims_or_calculations"],2)

    def test_partial_generation_run_is_not_evaluated(self):
        with self.assertRaisesRegex(ValueError,"complete, unique"):
            self.prepare([{"question_id":"Q1","configuration":"A","status":"generated"}])


if __name__=="__main__":
    unittest.main()

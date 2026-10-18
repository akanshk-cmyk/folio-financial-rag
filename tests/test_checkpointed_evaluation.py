"""Context: verify paid judge work is reusable without mixing changed inputs."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class CheckpointedEvaluationTests(unittest.TestCase):
    def module(self):
        path=Path(__file__).resolve().parents[1]/"examples/evaluate_checkpointed.py"
        spec=importlib.util.spec_from_file_location("checkpointed_judging",path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def fake_judge(self,source,destination,model,provider):
        data=json.loads(source.read_text())
        row=data["rows"][0]
        destination.write_text(json.dumps({"judge_provider":provider,"judge_model":model,
            "rows":[{"question_id":row["question_id"],"configuration":row["configuration"],
                     "faithfulness":1.0,"llm_context_precision_with_reference":0.5,"context_recall":0.5}]}))

    def test_resume_reuses_completed_scores_and_rejects_changed_input(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as folder:
            source,output=Path(folder)/"input.json",Path(folder)/"ragas.json"
            data={"manifest":{},"rows":[{"question_id":"Q1","configuration":"A","status":"generated"}]}
            source.write_text(json.dumps(data))
            with patch.object(module,"load_environment"),patch.object(module,"evaluate_ragas",side_effect=self.fake_judge) as judge,redirect_stdout(io.StringIO()):
                module.run(source,output,judge_model="test-judge")
                module.run(source,output,judge_model="test-judge")
                self.assertEqual(judge.call_count,1)
                data["manifest"]["changed"]=True
                source.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError,"Checkpoint settings changed"):
                    module.run(source,output,judge_model="test-judge")
            self.assertEqual(json.loads(output.read_text())["completed_responses"],1)

    def test_failure_keeps_earlier_results_and_can_resume(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as folder:
            source,output=Path(folder)/"input.json",Path(folder)/"ragas.json"
            source.write_text(json.dumps({"manifest":{},"rows":[
                {"question_id":"Q1","configuration":"A","status":"generated"},
                {"question_id":"Q2","configuration":"A","status":"generated"}]}))
            def fail_second(src,dst,model,provider):
                if json.loads(src.read_text())["rows"][0]["question_id"]=="Q2":
                    raise ValueError("simulated judge formatting failure")
                self.fake_judge(src,dst,model,provider)
            with patch.object(module,"load_environment"),patch.object(module,"evaluate_ragas",side_effect=fail_second),redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,"earlier scores are retained"):
                    module.run(source,output,judge_model="test-judge")
            self.assertEqual(json.loads(output.read_text())["completed_responses"],1)
            with patch.object(module,"load_environment"),patch.object(module,"evaluate_ragas",side_effect=self.fake_judge) as judge,redirect_stdout(io.StringIO()):
                module.run(source,output,judge_model="test-judge")
            self.assertEqual(judge.call_count,1)
            self.assertEqual(json.loads(output.read_text())["completed_responses"],2)


if __name__=="__main__":
    unittest.main()

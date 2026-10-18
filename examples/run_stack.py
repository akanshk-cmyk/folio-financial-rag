"""Context: end-to-end runner for the custom LangChain/Pinecone/Groq/RAGAS stack.

Downloads models and uploads vectors unless --skip-upload is selected. Groq-backed
evaluation is explicit via --evaluate. No namespaces are deleted or scores invented.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    """Context: stop at a failed stage and retain a reviewable execution log."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--questions", default="examples/apple_verified_questions.json")
    parser.add_argument("--model")
    args = parser.parse_args()
    stages = []
    if not args.skip_upload:
        stages.append(("build-and-upload", [sys.executable, "examples/upload_pinecone.py", "--create"]))
    generation = [sys.executable, "-m", "financial_rag", "experiment", "--fixed", "data/fixed.json",
                  "--semantic", "data/semantic.json", "--questions", args.questions,
                  "--provider", "groq", "--retrieval", "pinecone", "--output", "runs/pinecone-experiment.json"]
    if args.model:
        generation += ["--model", args.model]
    stages.append(("pinecone-retrieval-and-groq-generation", generation))
    if args.evaluate:
        stages.append(("ragas-judging", [sys.executable, "-m", "financial_rag", "evaluate", "--run",
                                         "runs/pinecone-experiment.json", "--output", "runs/pinecone-evaluation.json"]))
    Path("runs").mkdir(exist_ok=True)
    completed = []
    environment = {**os.environ, "PYTHONUTF8": "1", "RAGAS_DO_NOT_TRACK": "true"}
    for name, command in stages:
        print(f"Starting stage: {name}", flush=True)
        result = subprocess.run(command, env=environment, check=False)
        completed.append({"stage": name, "exit_code": result.returncode})
        Path("runs/stack-run.json").write_text(json.dumps(completed, indent=2), encoding="utf-8")
        if result.returncode:
            print(f"Stopped at {name}; review the error and runs/stack-run.json.", flush=True)
            return result.returncode
    print("Requested stages completed; inspect saved results before drawing quality conclusions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

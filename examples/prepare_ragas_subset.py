"""Context: retain a matched question subset for judging without accepting failed answers.

Every generation case stays in experiment.json. If any configuration fails on a
question, that question is excluded from ALL three judge configurations. This
produces explicitly partial RAGAS results and an auditable coverage record.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path


def main(folder=None):
    """Context: separate generation reliability from RAGAS scores on eligible responses."""
    folder = Path(folder) if folder else Path(__file__).resolve().parents[1]/"runs/apple-pinecone-benchmark-v1"
    source = folder/"experiment.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    questions = json.loads((folder/"questions.json").read_text(encoding="utf-8"))
    expected = {(q["id"], config) for q in questions for config in ("A", "B", "C")}
    cases = {(row["question_id"], row["configuration"]): row for row in data["rows"]}
    if len(cases) != len(data["rows"]) or set(cases) != expected:
        raise ValueError("Wait for the complete, unique 45-case generation run before preparing judging.")
    eligible = [q["id"] for q in questions if all(
        cases[(q["id"], config)]["status"] == "generated" for config in ("A", "B", "C"))]
    if not eligible:
        raise ValueError("No questions have generated responses in all three configurations.")
    coverage = {"planned_questions_per_configuration": len(questions),
        "eligible_questions_per_configuration": len(eligible), "eligible_question_ids": eligible,
        "excluded_question_ids": [q["id"] for q in questions if q["id"] not in eligible],
        "policy": "Matched subset: exclude a question from all configurations if any response is not generated.",
        "generation_cases": len(cases), "generation_statuses": dict(Counter(r["status"] for r in cases.values())),
        "failures": [{"question_id": r["question_id"], "configuration": r["configuration"],
                      "status": r["status"], "validation": r.get("validation"), "error": r.get("error")}
                     for r in cases.values() if r["status"] != "generated"],
        "response_coverage": {config: {"generated": sum(r["status"]=="generated" for r in cases.values() if r["configuration"]==config),
            "with_claims_or_calculations": sum(bool(r.get("claims") or r.get("calculations")) for r in cases.values() if r["configuration"]==config),
            "generated_without_claims_or_calculations": sum(r["status"]=="generated" and not(r.get("claims") or r.get("calculations")) for r in cases.values() if r["configuration"]==config)}
            for config in ("A", "B", "C")}}
    manifest = {**data["manifest"], "generation_file": source.name,
                "generation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "evaluation_coverage": coverage}
    (folder/"ragas-eligible-run.json").write_text(json.dumps({"manifest": manifest,
        "rows": [row for row in data["rows"] if row["question_id"] in eligible]}, indent=2), encoding="utf-8")
    (folder/"evaluation-coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    print(json.dumps(coverage, indent=2), flush=True)


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder",type=Path,help="Separate evaluation artifact directory")
    main(parser.parse_args().folder)

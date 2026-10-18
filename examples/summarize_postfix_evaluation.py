"""Context: summarize new measurements and compare only common scored questions.

Full post-fix scores cover all eligible questions. Historical scores may cover a
smaller subset, so aligned before/after means are computed separately and never
presented as if their original denominators were equal.
"""

from collections import Counter
import json
import math
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/"runs/apple-new-set-postfix-2026-10-18"
PREVIOUS=ROOT/"runs/apple-new-set-2026-10-18"
METRICS=("faithfulness","llm_context_precision_with_reference","context_recall")


def means(rows):
    """Context: average only finite observations and retain their exact denominator."""
    result={}
    for metric in METRICS:
        values=[row[metric] for row in rows if isinstance(row.get(metric),(int,float)) and math.isfinite(row[metric])]
        result[metric]={"mean":sum(values)/len(values) if values else None,"scored_cases":len(values)}
    return result


def main():
    """Context: verify completion, scope and shared candidate pools before publishing."""
    data=json.loads((FOLDER/"experiment.json").read_text(encoding="utf-8"))
    evaluation=json.loads((FOLDER/"ragas.json").read_text(encoding="utf-8"))
    questions=json.loads((FOLDER/"questions.json").read_text(encoding="utf-8"))
    previous=json.loads((PREVIOUS/"ragas.json").read_text(encoding="utf-8"))
    expected={(q["id"],c) for q in questions for c in ("A","B","C")}
    cases={(row["question_id"],row["configuration"]):row for row in data["rows"]}
    judged={(row["question_id"],row["configuration"]):row for row in evaluation["rows"]}
    if set(cases)!=expected or len(cases)!=len(data["rows"]):
        raise ValueError("Generation cases incomplete or duplicated.")
    if evaluation["completed_responses"]!=evaluation["planned_responses"] or len(judged)!=len(evaluation["rows"]):
        raise ValueError("RAGAS judging incomplete or duplicated.")
    shared=all(cases[(q["id"],"B")]["candidate_pool"]==cases[(q["id"],"C")]["candidate_pool"] for q in questions)
    if not shared:
        raise ValueError("B/C candidate pools differ.")
    previous_pairs={(r["question_id"],r["configuration"]) for r in previous["rows"]}
    common_ids=sorted({q["id"] for q in questions if all((q["id"],c) in judged and
        (q["id"],c) in previous_pairs for c in ("A","B","C"))})
    counts=Counter(row["status"] for row in data["rows"])
    findings=sum(row["status"]=="generated" and bool(row.get("claims") or row.get("calculations")) for row in data["rows"])
    refusals=sum(row["status"]=="generated" and not(row.get("claims") or row.get("calculations")) for row in data["rows"])
    summary={"questions":len(questions),"generation_cases":len(cases),"judged_responses":len(judged),
        "statuses":dict(counts),"responses_with_findings":findings,"insufficient_evidence_responses":refusals,
        "identical_B_C_candidate_pools":shared,"latest_scores":evaluation["summary"],
        "common_question_ids_for_historical_comparison":common_ids,
        "common_subset_comparison":{c:{"before":means([r for r in previous["rows"] if r["configuration"]==c and r["question_id"] in common_ids]),
            "after":means([r for r in evaluation["rows"] if r["configuration"]==c and r["question_id"] in common_ids])}
            for c in ("A","B","C")},
        "caveat":"RAGAS is LLM-judged, not financial answer accuracy. Routing, parser and calculator changes are shared across the corrected configurations. Full-run and historical matched-subset denominators differ."}
    (FOLDER/"postfix-summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()

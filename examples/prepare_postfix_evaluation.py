"""Context: freeze a fresh full evaluation after the validated statement-tool fixes.

Original questions and source files are verified and reused; old answers and
scores are never copied as new measurements. No API calls occur in this stage.
"""

import hashlib
import json
from pathlib import Path

from financial_rag.generation import load_environment,nebius_base_url,resolve_model


ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"runs/apple-pinecone-benchmark-v1"
SOURCE=ROOT/"runs/apple-new-set-2026-10-18"
OUT=ROOT/"runs/apple-new-set-postfix-2026-10-18"


def digest(path):
    """Context: fingerprint exact input and implementation bytes for review."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path,data):
    """Context: resume identical preparation without replacing frozen inputs."""
    encoded=json.dumps(data,indent=2,ensure_ascii=True)+"\n"
    if path.exists() and path.read_text(encoding="utf-8")!=encoded:
        raise ValueError("Post-fix frozen artifact changed: "+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(encoded,encoding="utf-8")


def main():
    """Context: keep references outside generation while documenting the new pipeline."""
    load_environment(ROOT/".env")
    original=json.loads((SOURCE/"frozen-corpus.json").read_text(encoding="utf-8"))
    questions=json.loads((SOURCE/"questions.json").read_text(encoding="utf-8"))
    documents=json.loads((BASE/"documents.json").read_text(encoding="utf-8"))
    if len(questions)!=15 or len({q["id"] for q in questions})!=15:
        raise ValueError("Require the complete new fifteen-question set.")
    if digest(BASE/"documents.json")!=original["canonical_sha256"]:
        raise ValueError("Canonical source corpus changed.")
    by_id={d["id"]:d for d in documents}
    for question in questions:
        for source in question["reference_sources"]:
            if not any(s["location"]==source["location"] and source["text"]==s["text"]
                       for s in by_id[source["document_id"]]["sections"]):
                raise ValueError("Reference passage differs: "+question["id"])
    for source in original["corpus"]:
        if digest(SOURCE/source["file"])!=source["sha256"]:
            raise ValueError("Original filing changed.")
    for snapshot in original["reused_vector_indexes"].values():
        if digest(ROOT/snapshot["file"])!=snapshot["sha256"]:
            raise ValueError("Dense vector snapshot changed.")
    write_once(OUT/"questions.json",questions)
    manifest={**original,"evaluation_set":"Financial Document Intelligence — New Set — Full Post-fix Evaluation",
        "questions_sha256":digest(OUT/"questions.json"),
        "previous_evaluation":"../apple-new-set-2026-10-18",
        "generator_provider":"nebius","generator_model":resolve_model(provider="nebius"),
        "judge_provider":"nebius","judge_model":resolve_model(judge=True,provider="nebius"),
        "provider_base_url":nebius_base_url(),
        "known_calculator_limit":"Sum now supports distinct, source-backed components; unrecognized layouts still fail closed.",
        "postfix_features":["company/year-filtered remotely verified statement anchors",
            "source-parsed year columns and calculator inputs", "audited sum operation",
            "unambiguous original dated sentence citations","JSON-enforced checkpointed RAGAS"],
        "interpretation":"A/B/C include the corrected common statement routing/tool layer. Old scores are historical; denominators must be compared explicitly.",
        "code_fingerprints":{name:digest(ROOT/"financial_rag"/name) for name in
            ("generation.py","calculator.py","statement_tools.py","overview.py","retrieval.py","cloud.py",
             "chains.py","taxonomy.py","evaluation.py","chunking.py","embedding.py")},
        "uv_lock_sha256":digest(ROOT/"uv.lock")}
    write_once(OUT/"frozen-corpus.json",manifest)
    for name in ("build-summary.json","pinecone-fixed.json","pinecone-semantic.json"):
        write_once(OUT/name,json.loads((SOURCE/name).read_text(encoding="utf-8")))
    print(f"Verified {len(questions)} questions and original source passages; post-fix run: {OUT}")
    print("Generation and judge models: "+manifest["generator_model"]+" / "+manifest["judge_model"])


if __name__=="__main__":
    main()

"""Context: rerun only the four failed cases with corrected production code.

The original 45-case evaluation is immutable. This is a targeted regression run,
not a replacement A/B/C benchmark or updated RAGAS score. Gold answers are used
only for post-generation checks and never enter the generation prompt.
"""

import hashlib
import json
import os
from pathlib import Path
import time

from financial_rag.chains import build_rag_chain
from financial_rag.cloud import PineconeSearch
from financial_rag.core import Index
from financial_rag.embedding import Embedder
from financial_rag.generation import load_environment,resolve_model
from report_new_evaluation import checks


ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"runs/apple-new-set-2026-10-18"
OUT=ROOT/"runs/apple-new-set-validation-fix-v3-2026-10-18"


def main():
    """Context: real Pinecone retrieval and Nebius generation verify each concrete fix."""
    os.environ.setdefault("HF_HUB_OFFLINE","1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
    load_environment(ROOT/".env")
    original=json.loads((SOURCE/"experiment.json").read_text(encoding="utf-8"))
    questions={q["id"]:q for q in json.loads((SOURCE/"questions.json").read_text(encoding="utf-8"))}
    cases=[r for r in original["rows"] if r["status"]=="validation-failed"]
    index=Index.load(ROOT/"runs/apple-pinecone-benchmark-v1/semantic.json")
    embedder=Embedder(index.manifest["embedding_model"],index.manifest["embedding_revision"])
    model=resolve_model(provider="nebius")
    search=PineconeSearch("financial-rag","semantic")
    manifest={"purpose":"Four-case regression; not a new full evaluation or RAGAS score",
        "original_generation_sha256":hashlib.sha256((SOURCE/"experiment.json").read_bytes()).hexdigest(),
        "provider":"nebius","model":model,"index_manifest":index.manifest,
        "statement_routing":"metadata-scoped original statement anchors fetched and verified in Pinecone",
        "code_fingerprints":{name:hashlib.sha256((ROOT/"financial_rag"/name).read_bytes()).hexdigest()
            for name in ("generation.py","calculator.py","retrieval.py","cloud.py","chains.py","taxonomy.py","statement_tools.py","overview.py")}}
    OUT.mkdir(parents=True,exist_ok=True)
    output=OUT/"answers.json"
    rows=[]
    if output.exists():
        previous=json.loads(output.read_text(encoding="utf-8"))
        if previous["manifest"]!=manifest:
            raise ValueError("Regression code/settings changed; preserve this artifact and use a new output directory.")
        rows=previous["rows"]
    for old in cases:
        if any(r["question_id"]==old["question_id"] and r["configuration"]==old["configuration"] for r in rows):
            continue
        question=questions[old["question_id"]]
        started=time.perf_counter()
        result=build_rag_chain(index,embedder,question["filters"],candidates=10,top_n=3,
            rerank=old["configuration"]=="C",provider="nebius",model=model,search_backend=search
            ).invoke({"question":question["question"]})
        row={**result,"question_id":question["id"],"configuration":old["configuration"],
            "seconds":time.perf_counter()-started,"original_status":old["status"],
            "original_validation":old["validation"],"reference":question["reference"],
            "checks":checks(question,result)}
        rows.append(row)
        temporary=output.with_suffix(".tmp")
        temporary.write_text(json.dumps({"manifest":manifest,"rows":rows},indent=2),encoding="utf-8")
        temporary.replace(output)
        print(question["id"]+" / "+old["configuration"]+": "+result["status"]+"; "+json.dumps(row["checks"]),flush=True)
    print("Saved targeted regression: "+str(output),flush=True)


if __name__=="__main__":
    main()

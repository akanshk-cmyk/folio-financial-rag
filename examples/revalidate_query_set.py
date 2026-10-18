"""Context: replay saved Groq completions through corrected validation, with no API.

The result is explicitly labeled as postprocessing of saved model responses, not
a fresh model run. Original model responses and pre-fix results remain untouched.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from financial_rag.core import Index
from financial_rag.generation import finish_answer
from evaluate_query_set import diagnostics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("destination",type=Path)
    args=parser.parse_args()
    args.destination.mkdir(parents=True,exist_ok=True)
    data=json.loads((args.source/"answers.json").read_text())
    questions=json.loads((args.source/"verified-questions.json").read_text())
    by_id={item["id"]:item for item in questions}
    root=Path(__file__).resolve().parents[1]
    index=Index.load(root/"data/apple-2024-2025-lexical.json")
    rows=[]
    for original in data["rows"]:
        row=dict(original)
        if original.get("raw_response") and not original.get("validation", "").startswith("Groq response did not finish"):
            response=SimpleNamespace(id=original["response_id"],usage=None,choices=[SimpleNamespace(
                finish_reason="stop",message=SimpleNamespace(content=original["raw_response"]))])
            base={key:original[key] for key in ("question","provider","evidence","sources","context_chars","dropped_chunks")}
            row.update(finish_answer(base,response,original["model"]))
            row["usage"]=original.get("usage",{})
        row["revalidation"]={"original_status":original["status"],"original_validation":original.get("validation"),
            "fresh_model_request":False,"seconds_are_original_generation_latency":True}
        row["checks"]=diagnostics(by_id[row["question_id"]],row,index)
        rows.append(row)
        print(row["question_id"],row["status"],row["checks"]["expected_number_coverage"])
    manifest={**data["manifest"],"postprocessing":"literal-whitespace-table-header-v2",
        "response_origin":str(args.source),"fresh_model_requests":False}
    (args.destination/"answers.json").write_text(json.dumps({"manifest":manifest,"rows":rows},indent=2),encoding="utf-8")
    for name in ("verified-questions.json","retrieval-diagnostics.json"):
        (args.destination/name).write_text((args.source/name).read_text(),encoding="utf-8")


if __name__=="__main__":
    main()

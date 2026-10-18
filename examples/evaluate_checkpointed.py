"""Context: checkpoint each question/configuration's actual RAGAS scores.

This uses the same three metrics and sequential judge settings as evaluate_ragas.
The reference/evidence inputs, model and judge implementation are fingerprinted;
completed scores are reused only when all these inputs remain unchanged.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from financial_rag.evaluation import evaluate_ragas
from financial_rag.generation import load_environment, resolve_model, nebius_base_url


METRICS=("faithfulness","llm_context_precision_with_reference","context_recall")


def write(path,data):
    """Context: atomically save results before proceeding to the next paid request."""
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data,indent=2,default=str),encoding="utf-8")
    temporary.replace(path)


def run(run_path,output,provider="nebius",judge_model=None):
    """Context: never score rejected answers or silently mix judge configurations."""
    load_environment()
    model=resolve_model(judge_model,judge=True,provider=provider)
    data=json.loads(run_path.read_text(encoding="utf-8"))
    if not data["rows"] or any(row["status"]!="generated" for row in data["rows"]):
        raise ValueError("Require eligible generated responses; preserve failures in the full experiment.")
    folder=output.parent/(output.stem+"-checkpoints")
    module_path=Path(__file__).resolve().parents[1]/"financial_rag/evaluation.py"
    settings={"input_sha256":hashlib.sha256(run_path.read_bytes()).hexdigest(),
        "judge_provider":provider,"judge_model":model,
        "judge_base_url":nebius_base_url() if provider=="nebius" else None,
        "judge_response_format":"json_object" if provider=="nebius" else "provider-default",
        "evaluation_sha256":hashlib.sha256(module_path.read_bytes()).hexdigest(),
        "checkpointing_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metrics":list(METRICS),"max_workers":1,"bypass_n":True,"max_outer_attempts_per_case":2}
    manifest_path=folder/"settings.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8"))!=settings:
        raise ValueError("Checkpoint settings changed; use a new output path to avoid mixing scores.")
    write(manifest_path,settings)
    records=[]
    def checkpoint():
        """Context: aggregate finite observations only, with a visible scored denominator."""
        summary={}
        for label in dict.fromkeys(r["configuration"] for r in data["rows"]):
            group=[r for r in records if r["configuration"]==label]
            summary[label]={"questions":len(group),"finite_score_counts":{}}
            for metric in METRICS:
                values=[r[metric] for r in group if isinstance(r.get(metric),(int,float)) and math.isfinite(r[metric])]
                summary[label][metric]=sum(values)/len(values) if values else None
                summary[label]["finite_score_counts"][metric]=len(values)
        write(output,{"judge_provider":provider,"judge_model":model,
            "judge_base_url":settings["judge_base_url"],"embedding_model":None,
            "source_manifest":data["manifest"],"judge_settings":settings,
            "planned_responses":len(data["rows"]),"completed_responses":len(records),
            "summary":summary,"rows":records})
    for row in data["rows"]:
        identity=row["question_id"]+"-"+row["configuration"]
        if not identity.replace("-","").isalnum():
            raise ValueError("Unsafe checkpoint case identifier.")
        destination=folder/(identity+".json")
        if destination.exists():
            result=json.loads(destination.read_text(encoding="utf-8"))
            print(identity+": reused completed RAGAS scores",flush=True)
        else:
            source=folder/(identity+"-input.json")
            write(source,{"manifest":data["manifest"],"rows":[row]})
            failures=[]
            for attempt in range(2):
                try:
                    evaluate_ragas(source,destination,model,provider=provider)
                    result=json.loads(destination.read_text(encoding="utf-8"))
                    print(identity+": RAGAS scores saved",flush=True)
                    break
                except Exception as exc:
                    # Context: keep exception types, never arbitrary provider bodies or credentials.
                    failures.append({"attempt":attempt+1,"error_type":type(exc).__name__})
                    write(folder/(identity+"-failures.json"),failures)
                    print(identity+": judge attempt failed ("+type(exc).__name__+")",flush=True)
            else:
                checkpoint()
                raise RuntimeError(identity+": judging failed twice; earlier scores are retained for resume.")
        if result["judge_provider"]!=provider or result["judge_model"]!=model or len(result["rows"])!=1:
            raise ValueError("Checkpoint judge configuration is inconsistent.")
        records.extend(result["rows"])
        checkpoint()
    print(f"Completed {len(records)} checkpointed RAGAS responses: {output}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--provider",choices=("nebius","groq","openai"),default="nebius")
    parser.add_argument("--judge-model")
    args=parser.parse_args()
    run(args.run,args.output,args.provider,args.judge_model)

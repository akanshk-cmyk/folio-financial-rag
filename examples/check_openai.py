"""Context: expose safe OpenAI access/quota diagnostics, never keys or error bodies."""

import argparse
import json
from pathlib import Path
import re

from openai import APIError,OpenAI

from financial_rag.generation import load_environment,resolve_model


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--probe",action="store_true")
    args=parser.parse_args()
    load_environment()
    report={"model":resolve_model(provider="openai"),"provider":"openai"}
    try:
        with OpenAI(base_url="https://api.openai.com/v1",timeout=20,max_retries=0) as client:
            client.models.retrieve(report["model"])
            report["model_access"]=True
            if args.probe:
                result=client.chat.completions.create(model=report["model"],messages=[
                    {"role":"user","content":'Return JSON only: {"ok":true}'}],
                    response_format={"type":"json_object"},max_completion_tokens=64,temperature=0)
                report["completion_access"]=result.choices[0].finish_reason=="stop"
    except APIError as exc:
        body=getattr(exc,"body",{})
        error=body.get("error",body) if isinstance(body,dict) else {}
        code=error.get("code") if isinstance(error,dict) else None
        report["error"]={"status":getattr(exc,"status_code",None),"type":type(exc).__name__,
            "code":code if isinstance(code,str) and re.fullmatch(r"[\w.:-]{1,80}",code) else "unspecified"}
    print(json.dumps(report,indent=2),flush=True)
    folder=Path("runs/revised-evaluation-15-openai")
    folder.mkdir(parents=True,exist_ok=True)
    (folder/"access-diagnostic.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return 1 if "error" in report else 0


if __name__=="__main__":
    raise SystemExit(main())

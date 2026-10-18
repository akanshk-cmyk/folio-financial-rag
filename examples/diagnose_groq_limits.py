"""Context: diagnose a representative blocked request without printing credentials.

Only numeric limit fields and a coarse rate-limit category are printed. Source
passages are public filings; provider messages and organization identifiers are not.
"""

import argparse
import json
from pathlib import Path
import re

from groq import APIError, Groq

from financial_rag.core import Index
from financial_rag.demo_ui import run_local_question
from financial_rag.generation import SYSTEM,load_environment,resolve_model


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--completion-tokens",type=int,default=3000)
    args=parser.parse_args()
    load_environment()
    folder=Path("runs/revised-evaluation-15")
    q=json.loads((folder/"verified-questions.json").read_text())[3]
    preview=run_local_question(Index.load(Path("data/apple-2024-2025-lexical.json")),q["question"],"AAPL",q["fiscal_years"])
    try:
        response=Groq(timeout=20,max_retries=0).chat.completions.create(model=resolve_model(),
            messages=[{"role":"system","content":SYSTEM},{"role":"user","content":json.dumps({
                "retrieved_context":preview["evidence"],"active_user_query":q["question"]})}],
            response_format={"type":"json_object"},temperature=0,max_completion_tokens=args.completion_tokens,
            reasoning_effort="low")
        report={"ok":True,"usage":response.usage.model_dump() if response.usage else {},"finish_reason":response.choices[0].finish_reason}
    except APIError as exc:
        error=getattr(exc,"body",{})
        error=error.get("error",error) if isinstance(error,dict) else {}
        message=error.get("message","") if isinstance(error,dict) else ""
        report={"ok":False,"status":getattr(exc,"status_code",None),"type":type(exc).__name__,
            "request_too_large":"request too large" in message.lower(),
            "tokens_per_minute":"tokens per minute" in message.lower(),
            "tokens_per_day":"tokens per day" in message.lower()}
        for field in ["limit","used","requested"]:
            match=re.search(r"\b"+field+r"\s*[:=]?\s*([\d,.]+)",message,re.I)
            if match: report[field]=match[1]
        retry=exc.response.headers.get("retry-after") if getattr(exc,"response",None) is not None else None
        if retry and re.fullmatch(r"\d+(?:\.\d+)?",retry): report["retry_after_seconds"]=retry
    print(json.dumps(report,indent=2),flush=True)
    (folder/"limit-diagnostic.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()

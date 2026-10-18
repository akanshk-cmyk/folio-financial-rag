"""Context: fresh revised-set generation and checkpointed Groq-backed RAGAS.

Reference passages are verified locally and withheld from generation. Per-question
RAGAS checkpoints avoid paying to repeat completed judging after an interruption.
This run evaluates the installed local BM25 demo, not a four-way chunking study.
"""

import argparse
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from financial_rag.calculator import numeric_literals
from financial_rag.core import Index, load_documents, stable_id
from financial_rag.demo_ui import run_local_question
from financial_rag.evaluation import evaluate_ragas
from financial_rag.generation import SYSTEM, load_environment, resolve_model
from financial_rag.retrieval import RETRIEVAL_POLICY


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/revised-evaluation-15"


def references(documents):
    """Context: original filing/year/physical-page provenance accompanies every label."""
    texts=[
        "According to the FY2024 filing, what were Apple’s total net sales for fiscal year 2024, in USD millions?",
        "According to the FY2025 filing, what were Apple’s total net sales for fiscal year 2025, in USD millions?",
        "According to the FY2024 filing, what was Apple’s net income for fiscal year 2024, in USD millions?",
        "According to the FY2025 filing, what was Apple’s diluted earnings per share for fiscal year 2025, in USD per share?",
        "According to the FY2024 filing, what research and development expense did Apple report for fiscal year 2024, in USD millions?",
        "According to the FY2025 filing, what were Apple’s iPhone net sales for fiscal year 2025, in USD millions?",
        "According to the FY2024 filing, what were Apple’s Services net sales for fiscal year 2024, in USD millions?",
        "According to the FY2025 filing, what was Apple’s total company gross margin percentage for fiscal year 2025?",
        "According to the FY2024 filing, what were Apple’s Mac net sales for fiscal year 2024, in USD millions?",
        "According to the FY2025 filing, what were Apple’s net sales for the Wearables, Home and Accessories category for fiscal year 2025, in USD millions?",
        "Using fiscal year 2023 and fiscal year 2024 total net sales from the FY2024 filing, what were the absolute change in USD millions and the percentage change, rounded to two decimal places?",
        "Using fiscal year 2024 Services net sales from the FY2024 filing and fiscal year 2025 Services net sales from the FY2025 filing, what was the percentage growth in Services net sales, rounded to two decimal places? Cite both filings.",
        "According to the FY2024 filing, what were Apple’s research and development expense, selling, general and administrative expense, and total operating expenses for fiscal year 2024, in USD millions?",
        "According to the FY2025 consolidated statement of cash flows, what was Apple’s net cash provided by operating activities for fiscal year 2025, in USD millions?",
        "According to the Risk Factors section of the FY2024 filing, what were three distinct risks associated with Apple’s supply chain or outsourced manufacturing? Provide a supporting citation for each risk."]
    growth_sales=((Decimal("391035")-Decimal("383285"))/Decimal("383285")*100).quantize(Decimal(".01"),rounding=ROUND_HALF_UP)
    growth_services=((Decimal("109158")-Decimal("96169"))/Decimal("96169")*100).quantize(Decimal(".01"),rounding=ROUND_HALF_UP)
    labels=[
        ([(2024,[41,47])],"FY2024 total net sales: USD 391,035 million.",["391035"]),
        ([(2025,[41,49])],"FY2025 total net sales: USD 416,161 million.",["416161"]),
        ([(2024,[41,42,49])],"FY2024 net income: USD 93,736 million.",["93736"]),
        ([(2025,[41,49])],"FY2025 diluted earnings per share: USD 7.46 per share.",["7.46"]),
        ([(2024,[34,41])],"FY2024 research and development expense: USD 31,370 million.",["31370"]),
        ([(2025,[32,49])],"FY2025 iPhone net sales: USD 209,586 million.",["209586"]),
        ([(2024,[33,41,47])],"FY2024 Services net sales: USD 96,169 million.",["96169"]),
        ([(2025,[33])],"FY2025 total company gross margin percentage: 46.9%.",["46.9"]),
        ([(2024,[33,47])],"FY2024 Mac net sales: USD 29,984 million.",["29984"]),
        ([(2025,[32,49])],"FY2025 Wearables, Home and Accessories net sales: USD 35,686 million.",["35686"]),
        ([(2024,[32,33,41])],f"FY2024 total net sales of USD 391,035 million minus FY2023 total net sales of USD 383,285 million gives an absolute increase of USD 7,750 million. Percentage change = (391035 - 383285) / 383285 * 100 = {growth_sales}% rounded to two decimal places.",["7750",str(growth_sales)]),
        ([(2024,[33,47]),(2025,[32,49])],f"FY2024 Services net sales from the FY2024 filing were USD 96,169 million; FY2025 Services net sales from the FY2025 filing were USD 109,158 million. Growth = (109158 - 96169) / 96169 * 100 = {growth_services}% rounded to two decimal places. Cite both source filings.",[str(growth_services)]),
        ([(2024,[34,41])],"FY2024 research and development expense: USD 31,370 million; selling, general and administrative expense: USD 26,097 million; total operating expenses: USD 57,467 million.",["31370","26097","57467"]),
        ([(2025,[46])],"FY2025 cash generated by operating activities, the cash-flow statement's reported operating cash flow: USD 111,482 million.",["111482"]),
        ([(2024,[10,13])],"Three supported risks are: (1) dependence on concentrated outsourcing partners, many outside the US, reduces control over manufacturing and distribution; (2) single/limited-source components and shortages expose Apple to supply constraints and price increases; (3) trade restrictions, geopolitical conflict, natural disasters and public-health events can interrupt manufacturing and logistics, raise costs, or delay deliveries. Cite a supporting Risk Factors passage for each distinct risk.",[])]
    by_year={doc.fiscal_year:doc for doc in documents}
    rows=[]
    for i,(scopes,reference,expected) in enumerate(labels,1):
        sources=[]
        for year,pages in scopes:
            doc=by_year[year]
            for section in doc.sections:
                match=re.search(r"PDF Page (\d+)",section.location)
                if match and int(match[1]) in pages:
                    sources.append({"document_id":doc.id,"fiscal_year":year,"pdf_page":int(match[1]),
                        "section":section.section_path,"location":section.location,"text":section.text})
        actual=numeric_literals("\n".join(s["text"] for s in sources))
        inputs=(["391035","383285"] if i==11 else ["109158","96169"] if i==12 else expected)
        if not sources or not {Decimal(v) for v in inputs} <= actual:
            raise ValueError(f"Source verification failed for Q{i:02d}")
        rows.append({"id":f"Q{i:02d}","question":texts[i-1],"fiscal_years":[scope[0] for scope in scopes],
            "reference":reference,"expected_numbers":expected,"reference_sources":sources,
            "required_calculations":["difference","growth"] if i==11 else ["growth"] if i==12 else [],
            "require_both_filings":i==12,"minimum_risk_claims":3 if i==15 else 0})
    return rows


def checks(q,result,index):
    """Context: deterministic reference coverage is not a semantic accuracy metric."""
    chunks={chunk.id:chunk for chunk in index.chunks}
    values=numeric_literals(" ".join(claim["text"] for claim in result.get("claims",[])))
    for calc in result.get("calculations",[]):
        values.update((Decimal(calc["raw_output"]),Decimal(calc["display_output"])))
    operations={calc["operation"] for calc in result.get("calculations",[])}
    used={s["evidence_id"] for c in result.get("claims",[]) for s in c["sources"]}
    used|={s["evidence_id"] for c in result.get("calculations",[]) for s in c["inputs"]}
    cited_years={chunks[result["sources"][key]["chunk_id"]].fiscal_year for key in used}
    has_claims=bool(result.get("claims") or result.get("calculations"))
    return {"substantive":has_claims,"expected_number_coverage":
            ({Decimal(v) for v in q["expected_numbers"]}<=values) if q["expected_numbers"] else None,
        "required_calculator_operations_present":set(q["required_calculations"])<=operations,
        "cited_filing_years":sorted(cited_years),
        "both_required_filings_cited":{2024,2025}<=cited_years if q["require_both_filings"] else None,
        "risk_claim_count":len(result.get("claims",[])) if q["minimum_risk_claims"] else None,
        "numeric_coverage_is_semantic_accuracy":False}


def write(path,data):
    """Context: checkpoint each completed answer and judge task to local JSON."""
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,default=str),encoding="utf-8")


def report(data,questions):
    """Context: all 15 remain visible even if generation or individual judging fails."""
    rows=data["rows"]
    metric_names=["faithfulness","llm_context_precision_with_reference","context_recall"]
    judged=[]
    for row in rows:
        path=OUT/"judged"/(row["question_id"]+".json")
        if path.exists():
            judged.extend(json.loads(path.read_text())["rows"])
    averages={}
    for metric in metric_names:
        finite=[r[metric] for r in judged if isinstance(r.get(metric),(int,float)) and math.isfinite(r[metric])]
        averages[metric]={"mean":sum(finite)/len(finite) if finite else None,"scored_questions":len(finite)}
    summary={"total_questions":15,"completed_generation":len(rows),"statuses":dict(Counter(r["status"] for r in rows)),
        "substantive_answers":sum(r.get("checks",{}).get("substantive",False) for r in rows),
        "eligible_generated":sum(r["status"]=="generated" for r in rows),"judged_questions":len(judged),"metrics":averages}
    write(OUT/"summary.json",summary)
    by_id={r["question_id"]:r for r in rows}
    scores={r["question_id"]:r for r in judged}
    lines=["# Revised 15-question RAGAS evaluation","",
        f"Fresh answers from the corrected local BM25 + {data['manifest'].get('provider','groq')} pipeline; no FlashRank or fixed/semantic comparison. "
        "Generation uses no references. RAGAS sees each response and its exact retained passages, plus the separately verified reference answer. "
        "The generator and judge models, prompts and code fingerprints are recorded in answers.json. "
        "Reference passages are included in verified-questions.json for independent review.","",
        "```json",json.dumps(summary,indent=2),"```","",
        "Context precision uses RAGAS LLMContextPrecisionWithReference; it is not BM25 precision or exact numeric accuracy. "
        "Refusals remain in the generation totals and may be judged. Validation/request failures are excluded from RAGAS; "
        "the score denominator is shown separately. Missing or failed scores are not replaced with zero.","",
        "| Question | Status | Reference | Number coverage | Faithfulness | Context precision | Context recall |",
        "|---|---|---|---|---|---|---|"]
    diagnostic=OUT/"limit-diagnostic.json"
    if diagnostic.exists():
        lines[2:2]=["Groq limit diagnostic: `"+json.dumps(json.loads(diagnostic.read_text()))+"`. "
            "No metric scores are available when judging is blocked. Failed provider requests are not incorrect-answer or zero-metric scores.",""]
    access=OUT/"access-diagnostic.json"
    if access.exists():
        lines[2:2]=["Provider access diagnostic (earlier attempts may be superseded by later successful requests): `"
            +json.dumps(json.loads(access.read_text()))+"`. No scores are inferred from blocked requests.",""]
    for q in questions:
        row=by_id.get(q["id"],{})
        s=scores.get(q["id"],{})
        def score(name):
            value=s.get(name)
            return f"{value:.3f}" if isinstance(value,(int,float)) and math.isfinite(value) else "unavailable"
        coverage=row.get("checks",{}).get("expected_number_coverage")
        lines.append(f"| {q['id']} | {row.get('status','not run')} | {q['reference']} | {coverage} | "
                     f"{score(metric_names[0])} | {score(metric_names[1])} | {score(metric_names[2])} |")
    lines.extend(["","## Answers and calculation/citation checks",""])
    for q in questions:
        row=by_id.get(q["id"],{})
        lines.extend([f"### {q['id']}","",q["question"],"",row.get("answer","Not generated"),"",
            "Checks: `"+json.dumps(row.get("checks",{}))+"`","",
            "Verified source pages: "+", ".join(sorted({f"FY{s['fiscal_year']} PDF {s['pdf_page']}" for s in q["reference_sources"]})),""])
    (OUT/"evaluation-report.md").write_text("\n".join(lines),encoding="utf-8")
    return summary


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument("--generate",action="store_true")
    parser.add_argument("--judge",action="store_true")
    parser.add_argument("--retry-failed",action="store_true",help="Explicitly retry request failures; preserve prior failures in the audit.")
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--provider",choices=["nebius","groq","openai"],default="nebius")
    parser.add_argument("--generator-model")
    parser.add_argument("--judge-model")
    args=parser.parse_args()
    OUT=args.output_dir or (ROOT/f"runs/revised-evaluation-15-{args.provider}" if args.provider!="groq" else OUT)
    load_environment(ROOT/".env")
    if args.generator_model:
        os.environ[args.provider.upper()+"_MODEL"]=args.generator_model
    if args.judge_model:
        os.environ[args.provider.upper()+"_JUDGE_MODEL"]=args.judge_model
    if (args.generate or args.judge) and not os.getenv(args.provider.upper()+"_API_KEY"):
        raise ValueError(f"Set {args.provider.upper()}_API_KEY locally before running this provider.")
    questions=references(load_documents(ROOT/"data/documents.json"))
    write(OUT/"verified-questions.json",questions)
    write(ROOT/"examples/apple_evaluation_revised_15.json",[{k:v for k,v in q.items() if k!="reference_sources"} for q in questions])
    fingerprints={name:hashlib.sha256((ROOT/"financial_rag"/name).read_bytes()).hexdigest()
                  for name in ["generation.py","calculator.py","retrieval.py","chains.py"]}
    manifest={"configuration":"corrected-local-bm25-no-rerank","provider":args.provider,
        "generator":resolve_model(provider=args.provider),
        "judge":resolve_model(judge=True,provider=args.provider),"candidates":10,"top_n":3,"context_chars":24000,
        "retrieval_policy":RETRIEVAL_POLICY,"prompt_hash":stable_id(SYSTEM),"code_fingerprints":fingerprints,
        "questions_hash":stable_id(json.dumps(questions,sort_keys=True))}
    index=Index.load(ROOT/"data/apple-2024-2025-lexical.json")
    manifest["index"]=index.manifest
    data=json.loads((OUT/"answers.json").read_text()) if (OUT/"answers.json").exists() else {"manifest":manifest,"rows":[]}
    if data["manifest"]!=manifest:
        raise ValueError("Existing revised-run configuration differs. Preserve it before starting a different run.")
    completed={row["question_id"] for row in data["rows"] if row["status"]!="request-failed" or not args.retry_failed}
    rate_limited=False
    if args.generate:
        for q in questions:
            if q["id"] in completed: continue
            started=time.perf_counter()
            try:
                result=run_local_question(index,q["question"],"AAPL",q["fiscal_years"],args.provider,model=manifest["generator"])
                row={**result,"checks":checks(q,result,index)}
            except Exception as exc:
                row={"question":q["question"],"answer":"","status":"request-failed","error_type":type(exc).__name__,"evidence":{}}
                # SDK/provider bodies can contain identifiers. Persist only our
                # transport's known sanitized status, never arbitrary error text.
                message=str(exc)
                if message.startswith(("Groq request failed (","OpenAI request failed (","Nebius request failed (")):
                    status=re.search(r"status (\d+)",message)
                    if status:
                        row["http_status"]=int(status[1])
                        rate_limited=row["http_status"]==429
            row.update(question_id=q["id"],configuration=manifest["configuration"],reference=q["reference"],seconds=time.perf_counter()-started)
            previous=next((r for r in data["rows"] if r["question_id"]==q["id"]),None)
            if previous:
                row["prior_request_failures"]=previous.get("prior_request_failures",[])+[
                    {key:previous.get(key) for key in ("status","error_type","http_status","seconds")}]
                data["rows"]=[r for r in data["rows"] if r["question_id"]!=q["id"]]
            data["rows"].append(row)
            data["rows"].sort(key=lambda r:r["question_id"])
            write(OUT/"answers.json",data)
            report(data,questions)
            print(q["id"],row["status"],row.get("checks",{}),flush=True)
            if rate_limited:
                print(f"{args.provider} rate limit reached; checkpointed and stopped rather than retrying remaining questions.",flush=True)
                break
    if args.judge and not rate_limited:
        failures={}
        for row in data["rows"]:
            if row["status"]!="generated": continue
            destination=OUT/"judged"/(row["question_id"]+".json")
            if destination.exists(): continue
            source=OUT/"judge-inputs"/(row["question_id"]+".json")
            write(source,{"manifest":manifest,"rows":[row]})
            try:
                evaluate_ragas(source,destination,manifest["judge"],provider=args.provider)
            except Exception as exc:
                failures[row["question_id"]]={"error_type":type(exc).__name__}
                write(OUT/"judge-errors.json",failures)
                print("Judge failed:",row["question_id"],type(exc).__name__,flush=True)
                if type(exc).__name__=="RateLimitError":
                    print(f"{args.provider} rate limit reached; stopping further judging. Resume after quota is available.",flush=True)
                    break
            report(data,questions)
            print("Judged:",row["question_id"],flush=True)
    print(json.dumps(report(data,questions),indent=2),flush=True)


if __name__=="__main__":
    main()

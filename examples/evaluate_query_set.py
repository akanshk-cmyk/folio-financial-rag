"""Context: evaluate the user's 15 questions against the actual local demo index.

References are source-checked, not sent to generation. Each answer is checkpointed
before judging. This is a BM25 baseline, not a substitute for the four-way study.
"""

import argparse
from decimal import Decimal
import json
from pathlib import Path
import re
import time

from financial_rag.calculator import numeric_literals
from financial_rag.core import Index, load_documents, stable_id
from financial_rag.demo_ui import run_local_question
from financial_rag.evaluation import evaluate_ragas
from financial_rag.generation import SYSTEM, load_environment, resolve_model
from financial_rag.retrieval import RETRIEVAL_POLICY, retrieve


ROOT = Path(__file__).resolve().parents[1]


def query_set(documents):
    """Context: label reporting period separately from the year of the source filing."""
    by_year = {doc.fiscal_year: doc for doc in documents}
    specs = [
        ("What was Apple's total net sales for fiscal year 2024?",2024,[41],"Total net sales were USD 391,035 million in FY2024.",["391035"]),
        ("What was Apple's total net sales for fiscal year 2025?",2025,[41],"Total net sales were USD 416,161 million in FY2025.",["416161"]),
        ("What was Apple's net income in fiscal year 2024?",2024,[41],"Net income was USD 93,736 million in FY2024.",["93736"]),
        ("What was Apple's diluted earnings per share for fiscal year 2025?",2025,[41],"Diluted earnings per share were USD 7.46 in FY2025.",["7.46"]),
        ("How much did Apple spend on research and development in fiscal year 2024?",2024,[41],"Research and development expense was USD 31,370 million in FY2024.",["31370"]),
        ("What was Apple's iPhone segment net sales in fiscal year 2024?",2024,[33,47],"iPhone category net sales were USD 201,183 million in FY2024. iPhone is a product category, not a reportable segment.",["201183"]),
        ("What was Apple's Services segment net sales in fiscal year 2024?",2024,[33,41,47],"Services category net sales were USD 96,169 million in FY2024. Services is a category, not a reportable segment.",["96169"]),
        ("What was Apple's gross margin percentage in fiscal year 2024?",2024,[34],"Total gross margin percentage was 46.2% in FY2024, distinct from Products margin of 37.2% and Services margin of 73.9%.",["46.2"]),
        ("What was Apple's total net sales for the Mac product line in fiscal year 2024?",2024,[33,47],"Mac net sales were USD 29,984 million in FY2024.",["29984"]),
        ("What was Apple's total net sales for the Wearables, Home and Accessories category in fiscal year 2024?",2024,[33,47],"Wearables, Home and Accessories net sales were USD 37,005 million in FY2024.",["37005"]),
        ("How did Apple's total net sales change between fiscal year 2023 and fiscal year 2024?",2024,[33,41],"Total net sales increased from USD 383,285 million in FY2023 to USD 391,035 million in FY2024, an increase of USD 7,750 million. The filing reports growth rounded to 2%.",["383285","391035","7750"]),
        ("What was Apple's total net sales for fiscal year 2023 (as reported in the FY2024 filing's comparative table)?",2024,[33,41],"FY2023 total net sales were USD 383,285 million in the FY2024 filing's comparative table.",["383285"]),
        ("What operating expenses did Apple report in fiscal year 2024?",2024,[34,41],"FY2024 operating expenses were research and development of USD 31,370 million and selling, general and administrative expenses of USD 26,097 million, totaling USD 57,467 million.",["31370","26097","57467"]),
        ("What was Apple's cash and cash equivalents position at the end of fiscal year 2024?",2024,[43],"Cash and cash equivalents were USD 29,943 million as of September 28, 2024.",["29943"]),
        ("What risk factors did Apple disclose related to supply chain in the FY2024 filing?",2024,[10,13],"Apple disclosed concentrated manufacturing and reliance on outsourcing partners, including single or limited sources; shortages and component price fluctuations; trade restrictions and geopolitical conflict; and natural disasters, public health events and other disruptions. These risks can delay supply and delivery, raise costs, and reduce production control.",[]),
    ]
    questions = []
    for number, (text,year,pages,reference,values) in enumerate(specs,1):
        doc = by_year[year]
        quotations = []
        for section in doc.sections:
            page = re.search(r"PDF Page (\d+)",section.location)
            if page and int(page[1]) in pages:
                quotations.append({"document_id":doc.id,"location":section.location,"text":section.text})
        if not quotations:
            raise ValueError(f"Reference source pages missing for question {number}")
        quoted_numbers = numeric_literals("\n".join(item["text"] for item in quotations))
        # Q11's delta is calculated independently, rather than asserted as a disclosure.
        source_values = values if number != 11 else values[:2]
        if not {Decimal(value) for value in source_values} <= quoted_numbers:
            raise ValueError(f"Reference values absent from designated source pages: Q{number}")
        questions.append({"id":f"Q{number:02d}","question":text,"reference":reference,
            "expected_numbers":values,"source_filing_year":year,"reporting_year":2023 if number==12 else year,
            "filters":{"ticker":"AAPL","fiscal_year":[year],"doc_type":"10-K"},
            "acceptable_pdf_pages":pages,"reference_sources":quotations})
    delta = Decimal("391035")-Decimal("383285")
    questions[10]["reference_calculation"] = {"formula":"(391035 - 383285) / 383285 * 100",
        "increase_usd_millions":str(delta),"growth_percent":str(delta/Decimal("383285")*100)}
    return questions


def diagnostics(question, result, index):
    """Context: retrieval and numeric diagnostics are separate from semantic accuracy."""
    chunks = {chunk.id:chunk for chunk in index.chunks}
    pages = []
    for source in result.get("sources",{}).values():
        match = re.search(r"PDF Page (\d+)",chunks[source["chunk_id"]].location)
        if match:
            pages.append(int(match[1]))
    numbers = numeric_literals("\n".join(claim["text"] for claim in result.get("claims",[])))
    for calc in result.get("calculations",[]):
        numbers.update([Decimal(calc["raw_output"]),Decimal(calc["display_output"])])
    return {"labeled_reference_page_hit_at_3":bool(set(pages)&set(question["acceptable_pdf_pages"])),
        "retained_pdf_pages":pages,
        "expected_number_coverage": ({Decimal(value) for value in question["expected_numbers"]} <= numbers)
                                    if question["expected_numbers"] else None,
        "numeric_coverage_is_semantic_accuracy":False,
        "validation_passed":result.get("validation")=="structural-and-numeric-checks-passed",
        "calculation_tool_used":bool(result.get("calculations"))}


def main():
    """Context: generation and RAGAS are separately requested; interrupted runs resume."""
    parser=argparse.ArgumentParser()
    parser.add_argument("--generate",action="store_true")
    parser.add_argument("--judge",action="store_true")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"runs/query-set-15")
    parser.add_argument("--question-id",action="append",help="Run selected IDs only, such as Q01; repeat for multiple IDs.")
    args=parser.parse_args()
    load_environment(ROOT/".env")
    folder=args.output_dir
    folder.mkdir(parents=True,exist_ok=True)
    questions=query_set(load_documents(ROOT/"data/documents.json"))
    if args.question_id:
        if set(args.question_id)-{q["id"] for q in questions}:
            raise ValueError("Unknown question ID.")
        questions=[q for q in questions if q["id"] in args.question_id]
    (folder/"verified-questions.json").write_text(json.dumps(questions,indent=2),encoding="utf-8")
    index=Index.load(ROOT/"data/apple-2024-2025-lexical.json")
    retrieval_rows=[]
    for question in questions:
        hits,_=retrieve(index,question["question"],filters=question["filters"],candidates=10,top_n=3,expand=True)
        pages=[int(re.search(r"PDF Page (\d+)",hit.chunk.location)[1]) for hit in hits]
        retrieval_rows.append({"id":question["id"],"retained_pdf_pages":pages,
            "labeled_reference_page_hit_at_3":bool(set(pages)&set(question["acceptable_pdf_pages"])),
            "source_page_labels_are_non_exhaustive":True})
    (folder/"retrieval-diagnostics.json").write_text(json.dumps(retrieval_rows,indent=2),encoding="utf-8")
    output=folder/"answers.json"
    manifest={"configuration":"local-bm25-no-rerank", "model":resolve_model(), "top_n":3,
        "retrieval_policy": RETRIEVAL_POLICY,
        "candidates":10,"context_chars":24000,"index":index.manifest,"prompt_hash":stable_id(SYSTEM),
        "questions_hash":stable_id(json.dumps(questions,sort_keys=True))}
    data=json.loads(output.read_text()) if output.exists() else {"manifest":manifest,"rows":[]}
    if data["manifest"]!=manifest:
        raise ValueError("Existing run settings differ; preserve or move its artifacts before rerunning.")
    completed={row["question_id"] for row in data["rows"]}
    if args.generate:
        for question in questions:
            if question["id"] in completed: continue
            started=time.perf_counter()
            try:
                result=run_local_question(index,question["question"],"AAPL",[question["source_filing_year"]],"groq")
                row={**result,"seconds":time.perf_counter()-started,
                    "checks":diagnostics(question,result,index)}
            except Exception as exc:
                # Never serialize a provider body, connection string or API key.
                row={"question":question["question"],"status":"request-failed","error_type":type(exc).__name__,
                     "answer":"","evidence":{},"seconds":time.perf_counter()-started}
            row.update(question_id=question["id"],configuration=manifest["configuration"],reference=question["reference"])
            data["rows"].append(row)
            output.write_text(json.dumps(data,indent=2),encoding="utf-8")
            print(question["id"],row["status"],row.get("checks",{}),flush=True)
    if args.judge:
        eligible=[row for row in data["rows"] if row["status"]=="generated"]
        if not eligible:
            raise ValueError("No generated answers are eligible for RAGAS.")
        judge_input=folder/"ragas-input.json"
        judge_input.write_text(json.dumps({"manifest":{**manifest,"eligible":len(eligible),
            "excluded":len(data["rows"])-len(eligible)},"rows":eligible},indent=2),encoding="utf-8")
        try:
            evaluate_ragas(judge_input,folder/"ragas-results.json")
        except Exception as exc:
            (folder/"judge-error.json").write_text(json.dumps({"type":type(exc).__name__,
                "note":"Judging did not finish; do not substitute or infer missing metric scores."}),encoding="utf-8")
            raise
    print("Artifacts:",folder,flush=True)


if __name__=="__main__":
    main()

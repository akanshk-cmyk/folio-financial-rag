"""Context: publish every new-set response, including failures and abstentions.

Numeric coverage, requested calculator operations and filing citations are useful
diagnostics, not a proof of semantic correctness. RAGAS results remain separate.
"""

from collections import Counter
import csv
from decimal import Decimal
from html import escape
import json
from pathlib import Path

from financial_rag.calculator import numeric_literals


ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/"runs/apple-new-set-2026-10-18"


def checks(question,row):
    """Context: inspect only verified claims/calculations, never rejected raw output."""
    claims=row.get("claims",[]) if row["status"]=="generated" else []
    calculations=row.get("calculations",[]) if row["status"]=="generated" else []
    values=numeric_literals(" ".join(claim["text"] for claim in claims))
    for calculation in calculations:
        values.update((Decimal(calculation["raw_output"]),Decimal(calculation["display_output"])))
    used={source["evidence_id"] for claim in claims for source in claim["sources"]}
    used|={item["evidence_id"] for calc in calculations for item in calc["inputs"]}
    years={row.get("sources",{}).get(key,{}).get("fiscal_year") for key in used}
    years.discard(None)
    return {"has_claims_or_calculations":bool(claims or calculations),
        "expected_number_coverage":({Decimal(n) for n in question["expected_numbers"]}<=values)
            if question["expected_numbers"] else None,
        "required_calculator_operations_present":set(question["required_calculations"])<=
            {calc["operation"] for calc in calculations},
        "cited_filing_years":sorted(years),
        "both_required_filings_cited":{2024,2025}<=years if question["require_both_filings"] else None,
        "numeric_coverage_is_semantic_accuracy":False}


def main(folder=None):
    """Context: save inspectable question-level results without changing the original run."""
    folder=Path(folder) if folder else FOLDER
    questions=json.loads((folder/"questions.json").read_text(encoding="utf-8"))
    experiment=json.loads((folder/"experiment.json").read_text(encoding="utf-8"))
    labels={"A":"Fixed", "B":"Semantic", "C":"Semantic + FlashRank"}
    by_id={q["id"]:q for q in questions}
    rows=[{**row,"checks":checks(by_id[row["question_id"]],row)} for row in experiment["rows"]]
    summary={config:{"cases":sum(r["configuration"]==config for r in rows),
        "statuses":dict(Counter(r["status"] for r in rows if r["configuration"]==config)),
        "with_claims_or_calculations":sum(r["checks"]["has_claims_or_calculations"] for r in rows if r["configuration"]==config),
        "numeric_reference_coverage":sum(r["checks"]["expected_number_coverage"] is True for r in rows if r["configuration"]==config),
        "numeric_reference_questions":sum(bool(q["expected_numbers"]) for q in questions)} for config in labels}
    (folder/"answer-checks.json").write_text(json.dumps({"summary":summary,
        "caveat":"Reference-number coverage is diagnostic only, not semantic answer accuracy.","rows":rows},indent=2),encoding="utf-8")
    with (folder/"answers.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        fields=["question_id","configuration","question","reference","status","answer","validation","checks"]
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row,"checks":json.dumps(row["checks"])})
    html=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Apple — new evaluation set</title>',
        '<style>body{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:20px;background:#faf8f5;color:#392d35}'
        'article{background:white;border:1px solid #ddd;padding:20px;margin:20px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}'
        'blockquote{border-left:3px solid #ad8a9b;padding-left:16px}a{color:#543043}</style>',
        '<h1>Financial Document Intelligence — New Set</h1><p><a href="report.html">RAGAS scores and coverage</a> · '
        '<a href="answers.csv">Download all responses as CSV</a></p>',
        '<p>All 45 cases are shown, including validation failures and refusals. The separately verified reference answers '
        'were withheld from generation. Numeric checks do not establish semantic correctness.</p>',
        '<h2>Response diagnostics</h2><pre>'+escape(json.dumps(summary,indent=2))+'</pre>']
    for question in questions:
        html+=['<article><h2>'+question["id"]+'</h2><p>'+escape(question["question"])+'</p>',
            '<p><strong>Verified reference:</strong> '+escape(question["reference"])+'</p>',
            '<p>Reference sources: '+escape('; '.join(f"FY{s['fiscal_year']}, {s['location']}" for s in question["reference_sources"]))+'</p>']
        for row in (r for r in rows if r["question_id"]==question["id"]):
            html+=['<h3>'+escape(labels[row["configuration"]])+': '+escape(row["status"])+'</h3>',
                '<pre>'+escape(row.get("answer",""))+'</pre>',
                '<details><summary>Validation and diagnostic checks</summary><pre>'+escape(json.dumps({
                    "validation":row.get("validation"),"checks":row["checks"]},indent=2))+'</pre></details>',
                ]
            if row["status"]=="validation-failed":
                # Context: rejected output is shown only as an explicitly labeled
                # audit artifact, never presented as a validated financial answer.
                html+=['<details><summary>Rejected model output — unverified, not an accepted answer</summary><pre>'
                    +escape(row.get("raw_response",""))+'</pre></details>']
            html+=['<details><summary>Retrieved evidence used for this response</summary>']
            for key,text in row.get("evidence",{}).items():
                source=row.get("sources",{}).get(key,{})
                html+=['<h4>'+escape(key+': '+source.get("citation",""))+'</h4><pre>'+escape(text)+'</pre>']
            html+=['</details>']
        html+=['</article>']
    html+=['</html>']
    (folder/"answers.html").write_text('\n'.join(html),encoding="utf-8")
    report=folder/"report.html"
    if report.exists():
        text=report.read_text(encoding="utf-8")
        link='<p><a href="answers.html">Review all 45 answers, verified references and original retrieved passages</a></p>'
        if link not in text:
            report.write_text(text.replace('</html>',link+'</html>'),encoding="utf-8")
    print(json.dumps(summary,indent=2))
    print("Question-level report: "+str(folder/"answers.html"))


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder",type=Path,help="Separate evaluation artifact directory")
    main(parser.parse_args().folder)

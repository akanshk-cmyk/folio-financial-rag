"""Context: summarize saved measurements without rerunning retrieval or generation."""

from collections import Counter
import json
from pathlib import Path
from statistics import mean
import argparse


def main():
    """Context: invalid outputs and refusals remain in the end-to-end denominator."""
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=root/"runs/query-set-15")
    folder=parser.parse_args().output_dir
    questions=json.loads((folder/"verified-questions.json").read_text())
    data=json.loads((folder/"answers.json").read_text())
    rows={row["question_id"]:row for row in data["rows"]}
    statuses=Counter(row["status"] for row in rows.values())
    numerical=[row for q in questions if q["expected_numbers"] and (row:=rows.get(q["id"]))]
    coverage=sum(row.get("checks",{}).get("expected_number_coverage") is True for row in numerical)
    substantive=sum(bool(row.get("claims") or row.get("calculations")) for row in rows.values())
    retrieval=json.loads((folder/"retrieval-diagnostics.json").read_text())
    lines=["# Evaluation: Apple financial query set (15 questions)","",
        "This evaluates the existing local Streamlit pipeline: structure-aware page chunks, "
        "BM25, explicit source-filing-year filters, 10 candidates, up to 3 passages, "
        "24,000-character context budget, no FlashRank, and Groq `openai/gpt-oss-120b`. "
        "References were withheld from generation. This is not a fixed-versus-semantic experiment.","",
        f"Completed questions: **{len(rows)}/{len(questions)}**. Status counts: `{dict(statuses)}`.","",
        f"Substantive validated answers (claims/calculations): **{substantive}/{len(questions)}**. "
        f"Expected-number coverage among the 14 numeric/multi-numeric questions: **{coverage}/{len(numerical)}**. "
        "This coverage diagnostic requires expected values in validated claims/calculator outputs; "
        "it is not a semantic correctness score and does not normalize equivalent billion/million units.","",
        f"Labeled reference page hit@3: **{sum(r['labeled_reference_page_hit_at_3'] for r in retrieval)}/{len(retrieval)}**. "
        "Reference-page labels are non-exhaustive. Another retrieved page may contain an equally valid answer; "
        "this is not complete evidence recall or RAGAS context recall.","",
        f"Mean end-to-end generation time: **{mean(row['seconds'] for row in rows.values()):.2f} seconds**. "
        "This includes provider retries/waiting; it is not retrieval latency.","",
        "## Verified references and actual outcomes","",
        "| ID | Verified reference | Source filing / PDF pages | Actual outcome | Number coverage |",
        "|---|---|---|---|---|"]
    if data["manifest"].get("postprocessing"):
        lines[2:2]=["This report revalidates saved Groq completions with corrected citation handling. "
            "It does not claim a new model request for each revalidated response. Original pre-fix and "
            "fresh-generation artifacts are preserved separately; timings are from the original generation.",""]
    for q in questions:
        row=rows.get(q["id"],{})
        covered=row.get("checks",{}).get("expected_number_coverage")
        coverage_label="n/a" if covered is None else "yes" if covered else "no"
        status=row.get("status","not run")
        if status=="generated" and not (row.get("claims") or row.get("calculations")):
            status="generated refusal"
        lines.append(f"| {q['id']} | {q['reference']} | FY{q['source_filing_year']} / {', '.join(map(str,q['acceptable_pdf_pages']))} | {status} | {coverage_label} |")
    lines.extend(["","## Calculation reference","",
        "Q11: USD 391,035 million minus USD 383,285 million equals USD 7,750 million. "
        f"Decimal-calculated growth is {questions[10]['reference_calculation']['growth_percent']}%; "
        "the filing's rounded reported growth is 2%. The model was required to use the constrained calculator for derived values.","",
        "## Query-set corrections","",
        "- Q6 and Q7 refer to product/service categories. Apple's reportable segments are geographic; "
        "the category values remain valid expected answers.",
        "- Q8 means consolidated total gross margin, 46.2%, rather than Products or Services margin.",
        "- Q11 accepts disclosed rounded growth; a calculated precise growth figure must have a tool audit. "
        "It does not require two separate documents: the FY2024 filing contains both years.",
        "- Q12 filters the FY2024 source filing while requesting its FY2023 comparative value.",
        "- Q13 expects both expense categories and their total; Q15 needs a narrative coverage rubric, "
        "not a single-number match.","","## Failure details",""])
    for q in questions:
        row=rows.get(q["id"],{})
        lines.extend([f"### {q['id']}","",f"Question: {q['question']}","",
            f"Returned answer: {row.get('answer','Not run')}","",
            f"Validation: {row.get('validation','not available')}","",
            f"Retrieved pages: {row.get('checks',{}).get('retained_pdf_pages',[])}.",""])
    ragas=folder/"ragas-results.json"
    if ragas.exists():
        judged=json.loads(ragas.read_text())
        lines.extend(["## RAGAS (eligible generated responses only)","",
            "```json",json.dumps(judged["summary"],indent=2),"```","",
            "Validation failures and request failures are excluded from RAGAS and remain in the "
            "end-to-end results above. Generated refusals may still be judged. High faithfulness "
            "does not establish a correct or complete answer. Missing/non-finite scores are not replaced.",""])
    else:
        lines.extend(["## RAGAS","","No completed RAGAS result file is available. No metric scores are inferred.",""])
    lines.extend(["## Review notes and next checks","",
        "1. Retrieve financial statement/table rows with their year and unit headers; inspect missing net-income and cash evidence.",
        "2. Align quote selection with validation: several plausible answers were rejected because short supporting quotes "
        "contained the metric but omitted the fiscal-year numeral used in the claim. Preserve validation and request sufficient quoted context.",
        "3. Add explicit refusal/answerability and narrative completeness checks alongside RAGAS.",
        "4. Rerun this same set after changes, then run the notebook's four controlled configurations with "
        "the same references. This baseline cannot establish a semantic-chunking or FlashRank winner.","",
        "Artifacts: `verified-questions.json`, `retrieval-diagnostics.json`, `answers.json`, "
        "and (when completed) `ragas-input.json` / `ragas-results.json`. The full source passages and "
        "raw rejected model responses are retained for review.",""])
    if data["manifest"].get("postprocessing"):
        start=lines.index("## Review notes and next checks")
        lines[start:]=["## Fixes applied and remaining checks","",
            "- Added financial numeric-row ranking to improve statement retrieval; complete page headers remain available.",
            "- Whitespace differences are accepted by recovering the original source span. Altered words, values and ellipses remain invalid.",
            "- Missing explicitly stated fiscal years can be supported by expanding a verified row quote to its nearest preceding table header. "
            "The original and expanded quotes are retained. A conflicting nearest header or a year-shaped monetary amount is not exempted.",
            "- Generation prompts request correct table columns, units, exact quotations, category terminology, and expense breakdowns.",
            "- Passing structural/numeric validation is not proof of correct financial interpretation. Review completeness, "
            "year-column alignment and unseen questions before claiming financial answer accuracy.",
            "- The old RAGAS scores are not copied to this report. Judge the updated responses separately if needed.",""]
    output=folder/"evaluation-report.md"
    output.write_text("\n".join(lines),encoding="utf-8")
    print(output)
    print(dict(statuses), "substantive",substantive,"numeric coverage",coverage,"/",len(numerical))


if __name__=="__main__":
    main()

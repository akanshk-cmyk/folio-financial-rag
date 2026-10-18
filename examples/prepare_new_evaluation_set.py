"""Context: freeze the user's new 15-question set with locally verified filing evidence.

Reference answers are evaluation labels, never supplied to generation. Existing
Apple PDFs, embeddings and verified Pinecone namespaces are reused unchanged.
Only new question, reference and result files are written to a separate directory.
"""

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re


ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"runs/apple-pinecone-benchmark-v1"
OUT=ROOT/"runs/apple-new-set-2026-10-18"


def digest(path):
    """Context: record exact bytes used for this evaluation without copying dense vectors."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path, content):
    """Context: a rerun may reuse identical inputs but never silently change references."""
    encoded=json.dumps(content,indent=2,ensure_ascii=True)+"\n"
    if path.exists() and path.read_text(encoding="utf-8")!=encoded:
        raise ValueError("Frozen input differs: "+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(encoded,encoding="utf-8")


def main():
    """Context: verify literal source passages and calculate labels with Decimal arithmetic."""
    documents=json.loads((BASE/"documents.json").read_text(encoding="utf-8"))
    by_year={d["fiscal_year"]:d for d in documents}
    def source(year,page,needles):
        document=by_year[year]
        sections=[s for s in document["sections"] if re.match(rf"PDF Page {page}(?:\s|$)",s["location"])]
        if len(sections)!=1:
            raise ValueError(f"Ambiguous source page {year}/{page}")
        section=sections[0]
        for needle in needles:
            if needle not in section["text"]:
                raise ValueError(f"Reference text absent: {year}/{page}/{needle}")
        return {"document_id":document["id"],"fiscal_year":year,"pdf_page":page,
                "section":section["section_path"],"location":section["location"],"text":section["text"]}

    human=source(2024,8,["164,000 full-time equivalent employees"])
    repurchases=source(2024,61,["499 million shares", "$95.0 billion"])
    balance24=source(2024,43,["33,410","7,286","September 28,"])
    balance25=source(2025,43,["39,777","5,718","69,860","7,979","12,350","78,328"])
    cash25=source(2025,46,["Depreciation and amortization","11,698","Share-based compensation expense","12,863","43,369","26,102"])
    revenue=source(2025,49,["$13.7 billion","expects 66%"])
    policies=source(2025,47,["Inventories are measured using the first-in, first-out method.",
        "Depreciation on property, plant and equipment is recognized on a straight-line basis."])
    dividend=source(2024,37,["$0.25 per share", "subject to declaration by the Board."])

    receivable_change=Decimal("39777")-Decimal("33410")
    receivable_growth=(receivable_change/Decimal("33410")*100).quantize(Decimal(".01"))
    inventory_change=Decimal("5718")-Decimal("7286")
    inventory_growth=(inventory_change/Decimal("7286")*100).quantize(Decimal(".01"))
    debt=Decimal("12350")+Decimal("78328")

    texts=[
        "According to the FY2024 filing’s Human Capital section, approximately how many full-time equivalent employees did Apple have as of September 28, 2024?",
        "According to the FY2024 filing’s Shareholders’ Equity note, how many shares did Apple repurchase during fiscal year 2024, and what was their total purchase cost?",
        "According to the FY2024 consolidated balance sheet, what were Apple’s net accounts receivable as of September 28, 2024, in USD millions?",
        "According to the FY2024 consolidated balance sheet, what were Apple’s inventories as of September 28, 2024, in USD millions?",
        "According to the FY2025 consolidated balance sheet, what were Apple’s accounts payable as of September 27, 2025, in USD millions?",
        "According to the FY2025 consolidated balance sheet, what was Apple’s outstanding commercial paper balance as of September 27, 2025, in USD millions?",
        "According to the FY2025 consolidated statement of cash flows, what depreciation and amortization adjustment did Apple report for fiscal year 2025, in USD millions?",
        "According to the FY2025 consolidated statement of cash flows, what share-based compensation expense did Apple report for fiscal year 2025, in USD millions?",
        "According to the FY2025 consolidated statement of cash flows, how much cash did Apple pay for income taxes, net, during fiscal years 2025 and 2024, in USD millions?",
        "According to the FY2025 Revenue note, what was Apple’s total deferred revenue as of September 27, 2025, in USD billions, and what percentage was expected to be realized within less than one year?",
        "Using net accounts receivable from the FY2024 and FY2025 consolidated balance sheets, what were the absolute increase in USD millions and percentage increase between the two fiscal year-end dates? Round the percentage to two decimal places and cite both filings.",
        "Using inventories from the FY2024 and FY2025 consolidated balance sheets, what were the absolute change in USD millions and percentage change between the two fiscal year-end dates? Round the percentage to two decimal places and cite both filings.",
        "According to the FY2025 consolidated balance sheet, what was Apple’s total carrying amount of term debt as of September 27, 2025, calculated by adding its current and non-current portions, in USD millions?",
        "According to the FY2025 Summary of Significant Accounting Policies, what inventory costing method does Apple use, and what depreciation method does it use for property, plant and equipment?",
        "According to the FY2024 filing’s Capital Return Program discussion, what was Apple’s quarterly cash dividend per share as of September 28, 2024, and what condition applied to its stated intention to increase the dividend annually?",
    ]
    references=[
        "Approximately 164,000 full-time equivalent employees as of September 28, 2024.",
        "During fiscal 2024 Apple repurchased 499 million common shares for a total purchase cost of USD 95.0 billion.",
        "Net accounts receivable at September 28, 2024: USD 33,410 million.",
        "Inventories at September 28, 2024: USD 7,286 million.",
        "Accounts payable at September 27, 2025: USD 69,860 million.",
        "Commercial paper on the FY2025 balance sheet: USD 7,979 million.",
        "FY2025 depreciation and amortization adjustment: USD 11,698 million.",
        "FY2025 share-based compensation expense adjustment: USD 12,863 million.",
        "Cash paid for income taxes, net: USD 43,369 million for FY2025 and USD 26,102 million for FY2024.",
        "Total deferred revenue at September 27, 2025: USD 13.7 billion; 66% was expected to be realized in less than a year.",
        f"Net accounts receivable increased from USD 33,410 million (FY2024 filing) to USD 39,777 million (FY2025 filing): increase USD {receivable_change} million; percentage increase {receivable_growth}%.",
        f"Inventories changed from USD 7,286 million (FY2024 filing) to USD 5,718 million (FY2025 filing): signed change USD {inventory_change} million (a decrease); percentage change {inventory_growth}%.",
        f"Term debt carrying amount: current USD 12,350 million plus non-current USD 78,328 million = USD {debt} million. This excludes commercial paper.",
        "Apple measures inventories using first-in, first-out (FIFO) and depreciates property, plant and equipment on a straight-line basis.",
        "Quarterly cash dividend: USD 0.25 per share as of September 28, 2024. Annual dividend increases were intended subject to declaration by the Board of Directors.",
    ]
    sources=[[human],[repurchases],[balance24],[balance24],[balance25],[balance25],[cash25],[cash25],
             [cash25],[revenue],[balance24,balance25],[balance24,balance25],[balance25],[policies],[dividend]]
    numbers=[["164000"],["499","95.0"],["33410"],["7286"],["69860"],["7979"],["11698"],["12863"],
             ["43369","26102"],["13.7","66"],[str(receivable_change),str(receivable_growth)],
             [str(inventory_change),str(inventory_growth)],[str(debt)],[],["0.25"]]
    questions=[]
    for i,(question,reference,refs,expected) in enumerate(zip(texts,references,sources,numbers),1):
        years=sorted({s["fiscal_year"] for s in refs})
        questions.append({"id":f"Q{i:02d}","question":question,"reference":reference,
            "fiscal_years":years,"filters":{"ticker":"AAPL","fiscal_year":years,"doc_type":"10-K"},
            "expected_numbers":expected,"reference_sources":refs,"require_both_filings":i in {11,12},
            "required_calculations":["difference","growth"] if i in {11,12} else ["sum"] if i==13 else []})
    write_once(OUT/"questions.json",questions)
    original=json.loads((BASE/"frozen-corpus.json").read_text(encoding="utf-8"))
    manifest={**original,"evaluation_set":"Financial Document Intelligence Pipeline — New Set",
        "canonical_file":"../apple-pinecone-benchmark-v1/documents.json",
        "questions_sha256":digest(OUT/"questions.json"),"reference_quotations_verified":sum(map(len,sources)),
        "reference_calculations":{"Q11":{"difference":str(receivable_change),"growth_percent":str(receivable_growth)},
            "Q12":{"difference":str(inventory_change),"growth_percent":str(inventory_growth)},"Q13":{"sum":str(debt)}},
        "reused_vector_indexes":{name:{"file":str((BASE/(name+".json")).relative_to(ROOT)),
            "sha256":digest(BASE/(name+".json"))} for name in ("fixed","semantic")},
        "known_calculator_limit":"Generation calculator has no sum operation; Q13 tests this limitation unchanged.",
        "uv_lock_sha256":digest(ROOT/"uv.lock")}
    for entry in manifest["corpus"]:
        if digest(BASE/entry["file"])!=entry["sha256"]:
            raise ValueError("Original filing hash changed.")
        entry["file"]="../apple-pinecone-benchmark-v1/"+entry["file"]
    write_once(OUT/"frozen-corpus.json",manifest)
    for name in ("build-summary.json","pinecone-fixed.json","pinecone-semantic.json"):
        artifact=json.loads((BASE/name).read_text(encoding="utf-8"))
        write_once(OUT/name,{**artifact,"reused_from":"../apple-pinecone-benchmark-v1/"+name})
    print(f"Verified {len(questions)} new references; reused both dense indexes. Output: {OUT}")
    print(json.dumps(manifest["reference_calculations"],indent=2))


if __name__=="__main__":
    main()

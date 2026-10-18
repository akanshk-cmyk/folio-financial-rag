"""Context: freeze Apple PDFs/references and build both benchmark vector snapshots.

Run from the project root. This stage uses cached model weights only and makes
no Pinecone or LLM requests. It never replaces the interactive demo's indexes.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from datetime import datetime, timezone
from decimal import Decimal
from importlib.metadata import version

from financial_rag.chunking import build_index
from financial_rag.core import Index, load_documents
from financial_rag.embedding import Embedder, MODEL
from financial_rag.ingest import save_documents


ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/"runs/apple-pinecone-benchmark-v1"
REVISION="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"


def digest(path):
    """Context: hash complete bytes so inputs can be checked before resuming."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, content):
    """Context: refuse to silently change a frozen artifact when resuming."""
    encoded=json.dumps(content,indent=2,ensure_ascii=True)+"\n"
    if path.exists() and path.read_text(encoding="utf-8")!=encoded:
        raise ValueError("Frozen artifact differs: "+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(encoded,encoding="utf-8")


def freeze():
    """Context: validate source hashes and reference quotations before embedding."""
    documents=sorted((d for d in load_documents(ROOT/"data/documents.json")
                      if d.ticker=="AAPL" and d.fiscal_year in {2024,2025}),key=lambda d:d.fiscal_year)
    if len(documents)!=2 or {d.fiscal_year for d in documents}!={2024,2025}:
        raise ValueError("Require exactly one Apple filing for each benchmark year.")
    FOLDER.mkdir(parents=True,exist_ok=True)
    raw=[]
    for document in documents:
        original=ROOT/f"data/raw/aapl-{document.fiscal_year}.pdf"
        if digest(original)!=document.content_hash:
            raise ValueError("PDF hash differs from canonical document: "+document.id)
        destination=FOLDER/"raw"/original.name
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists() and digest(destination)!=document.content_hash:
            raise ValueError("Frozen PDF was changed.")
        if not destination.exists():
            shutil.copyfile(original,destination)
        raw.append({"document_id":document.id,"fiscal_year":document.fiscal_year,
                    "file":str(destination.relative_to(FOLDER)),"sha256":document.content_hash,
                    "source":document.source,"sections":len(document.sections)})
    canonical=FOLDER/"documents.json"
    if canonical.exists():
        # Context: compare complete canonical records, not only filenames or years.
        from dataclasses import asdict
        if json.loads(canonical.read_text(encoding="utf-8"))!=[asdict(d) for d in documents]:
            raise ValueError("Frozen canonical corpus differs from current input.")
    else:
        save_documents(documents,canonical)
    questions=json.loads((ROOT/"runs/revised-evaluation-15/verified-questions.json").read_text(encoding="utf-8"))
    if len(questions)!=15 or len({q["id"] for q in questions})!=15:
        raise ValueError("Require the revised fifteen-question benchmark.")
    by_id={d.id:d for d in documents}
    for question in questions:
        for source in question["reference_sources"]:
            document=by_id[source["document_id"]]
            if not source.get("text") or not any(source["text"] in section.text and
                    section.location==source["location"] for section in document.sections):
                raise ValueError("Unverified reference quotation: "+question["id"])
        question["filters"]={"ticker":"AAPL","fiscal_year":question["fiscal_years"],"doc_type":"10-K"}
    write_once(FOLDER/"questions.json",questions)
    # Context: independently recalculate the two comparative reference answers.
    from financial_rag.overview import financial_overview
    source_index=build_index(documents,"enhanced")
    reports=financial_overview(source_index.chunks,"AAPL",[2024,2025])
    current=Decimal(reports[2024]["metrics"]["sales"]["value"])
    prior=Decimal(reports[2024]["metrics"]["sales"]["prior_value"])
    services_current=Decimal(reports[2025]["products"]["Services"]["value"])
    services_prior=Decimal(reports[2024]["products"]["Services"]["value"])
    calculations={"Q11":{"difference":str(current-prior),"growth_percent":str(((current-prior)/prior*100).quantize(Decimal('.01')))},
                  "Q12":{"growth_percent":str(((services_current-services_prior)/services_prior*100).quantize(Decimal('.01')))}}
    if calculations["Q11"]!={"difference":"7750","growth_percent":"2.02"} or calculations["Q12"]["growth_percent"]!="13.51":
        raise ValueError("Comparative reference arithmetic failed.")
    manifest={"corpus":raw,"canonical_sha256":digest(canonical),"questions_sha256":digest(FOLDER/"questions.json"),
              "questions":len(questions),"reference_quotations_verified":sum(len(q["reference_sources"]) for q in questions),
              "reference_calculations":calculations,"embedding_model":MODEL,"embedding_revision":REVISION,
              "fixed_size_characters":512,"fixed_overlap_characters":64,"semantic_percentile":65,
              "semantic_buffer_sentences":1,"candidates":10,"top_n":3,"adaptive":False,
              "input_format":"page-aware PDFs; not the walkthrough's full EDGAR submissions",
              "uv_lock_sha256":digest(ROOT/"uv.lock")}
    write_once(FOLDER/"frozen-corpus.json",manifest)
    print("Frozen: two PDFs, fifteen questions; all reference source quotations verified.",flush=True)
    return documents


def main():
    """Context: resume completed indexes; build missing indexes with one pinned encoder."""
    os.environ["HF_HUB_OFFLINE"]="1"
    os.environ["TRANSFORMERS_OFFLINE"]="1"
    documents=freeze()
    model=None
    summary_path=FOLDER/"build-summary.json"
    summary=json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {
        "created_at":datetime.now(timezone.utc).isoformat(),"indexes":{},
        "packages":{name:version(name) for name in ("sentence-transformers","torch","pinecone","langchain-experimental")}}
    for strategy in ("fixed","semantic"):
        path=FOLDER/f"{strategy}.json"
        started=time.perf_counter()
        if path.exists():
            previous=summary["indexes"].get(strategy,{})
            if previous.get("sha256") and digest(path)!=previous["sha256"]:
                raise ValueError("Stored vector snapshot was changed: "+strategy)
            index=Index.load(path)
            if (index.manifest.get("embedding_model")!=MODEL or index.manifest.get("embedding_revision")!=REVISION
                    or {item['id'] for item in index.manifest['corpus']}!={d.id for d in documents}):
                raise ValueError("Saved benchmark snapshot differs from frozen configuration.")
            print("Reusing "+strategy+" snapshot.",flush=True)
        else:
            if model is None:
                print("Loading cached BGE weights; external model downloads disabled.",flush=True)
                model=Embedder(MODEL,REVISION)
                print("Encoder ready on "+str(model.model.device),flush=True)
            print("Building "+strategy+" chunks and embeddings...",flush=True)
            index=build_index(documents,strategy,model)
            index.save(path)
        if len(index.vectors)!=len(index.chunks) or any(len(v)!=768 for v in index.vectors):
            raise ValueError("Incomplete or incompatible vector snapshot.")
        elapsed=summary["indexes"].get(strategy,{}).get("elapsed_seconds",time.perf_counter()-started)
        summary["indexes"][strategy]={"file":path.name,"sha256":digest(path),"chunks":len(index.chunks),
                                       "vectors":len(index.vectors),"dimension":768,
                                       "elapsed_seconds":elapsed}
        (FOLDER/"build-summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
        print(strategy+": "+str(len(index.vectors))+" vectors saved.",flush=True)
    print("Local preparation complete: "+str(FOLDER),flush=True)


if __name__=="__main__":
    main()

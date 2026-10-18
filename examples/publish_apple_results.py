"""Context: publish measured benchmark artifacts as local HTML/JSON, never sample scores."""

import json
import math
from pathlib import Path
from html import escape
from collections import Counter


def main(folder=None):
    """Context: distinguish completed connection/build steps from unrun evaluation stages."""
    folder=Path(folder) if folder else Path(__file__).resolve().parents[1]/"runs/apple-pinecone-benchmark-v1"
    def read(name):
        path=folder/name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    corpus=read("frozen-corpus.json")
    builds=read("build-summary.json")
    experiments=read("experiment.json")
    evaluation=read("ragas.json")
    coverage=read("evaluation-coverage.json")
    rows=experiments.get("rows",[])
    # Context: abstentions can yield NaN faithfulness. Show the scored denominator
    # so an average over very few substantive answers cannot imply full coverage.
    score_counts={label:{name:sum(isinstance(row.get(name),(int,float)) and math.isfinite(row[name])
        for row in evaluation.get("rows",[]) if row.get("configuration")==label)
        for name in ("faithfulness","llm_context_precision_with_reference","context_recall")}
        for label in ("A","B","C")}
    summary={"index":"financial-rag","evaluation_set":corpus.get("evaluation_set","Original Apple question set"),
             "corpus_documents":len(corpus.get("corpus",[])),
             "questions":corpus.get("questions",0),"source_quotes_verified":corpus.get("reference_quotations_verified",0),
             "builds":builds.get("indexes",{}),"uploads":{key:read("pinecone-"+key+".json") for key in ("fixed","semantic")},
             "retrieval_cases_recorded":len(read("retrieval-checks.json").get("rows",[])) or
                 sum(bool(row.get("candidate_pool")) for row in rows),
             "generation_statuses":dict(Counter(row["status"] for row in rows)),
             "generation_cases":len(rows),"ragas_summary":evaluation.get("summary",{}),
             "evaluation_coverage":coverage,"finite_score_counts":score_counts,
             "provider":experiments.get("manifest",{}).get("provider"),
             "generation_model":experiments.get("manifest",{}).get("model"),
             "judge_model":evaluation.get("judge_model"),
             "evaluation_status":("measured on matched eligible subset; not a full 15-question evaluation"
                  if coverage.get("excluded_question_ids") else "measured") if evaluation.get("rows") else "not measured",
             "results_are_from_pdf_corpus_not_full_edgar_walkthrough":True}
    (folder/"results-summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    cells=[]
    for key,title in (("fixed","A: Fixed"),("semantic","B/C: Semantic")):
        build=summary["builds"].get(key,{})
        upload=summary["uploads"][key]
        cells.append('<tr><td>'+escape(title)+'</td><td>'+str(build.get("vectors","Building"))+'</td><td>'
                     +escape(upload.get("verification","Pending"))+'</td><td>'
                     +str(upload.get("namespace_vector_count","Not verified"))+'</td></tr>')
    scores=[]
    for label in ("A","B","C"):
        metrics=summary["ragas_summary"].get(label,{})
        scores.append('<tr><td>'+label+'</td>'+''.join('<td>'+escape(str(metrics.get(name,"Not measured")))+
                      ' (n='+str(score_counts[label][name])+')</td>'
                      for name in ("faithfulness","llm_context_precision_with_reference","context_recall"))+'</tr>')
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Apple financial RAG benchmark</title>
    <style>body{font:16px/1.6 system-ui;max-width:1000px;margin:40px auto;padding:0 20px;background:#faf8f5;color:#392d35}
    h1{font:36px Georgia;color:#543043}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{padding:12px;text-align:left;border-bottom:1px solid #dfd3d9}
    th{background:#eee6eb}pre{white-space:pre-wrap;background:#f0e9ed;padding:18px}a{color:#543043}</style>
    <h1>Apple financial RAG benchmark</h1><p>Frozen FY2024/FY2025 PDFs; 15 questions; fixed 512/64 characters; semantic 65th percentile;
    pinned BGE 768-dimensional embeddings. Counts reflect these PDFs and are not copied from the EDGAR walkthrough.</p>
    <h2>Measured build and Pinecone storage</h2><table><tr><th>Dataset</th><th>Local vectors</th><th>Upload verification</th><th>Remote count</th></tr>'''
    html+=''.join(cells)+'</table><h2>RAGAS results</h2><p>'+escape(summary["evaluation_status"])+\
          '</p><table><tr><th>Configuration</th><th>Faithfulness</th><th>Context precision</th><th>Context recall</th></tr>'+''.join(scores)+'</table>'
    html+='<p>Generated cases recorded: '+str(len(rows))+'. Retrieval cases recorded: '+str(summary["retrieval_cases_recorded"])+'.</p>'
    html+='<p>Provider: '+escape(str(summary["provider"]))+'; generation model: '+escape(str(summary["generation_model"]))+'; judge model: '+escape(str(summary["judge_model"]))+'</p>'
    if builds.get("reused_from"):
        html+='<p>This evaluation reuses the original frozen Apple vector indexes and upload receipts. It did not rebuild or re-upload the corpus.</p>'
    html+='<h2>Coverage and limitations</h2><p>Metrics are LLM judgments, not proof of financial correctness. Each n is the number of finite scores; abstentions can have undefined faithfulness. No overall winner is inferred from faithfulness alone.</p>'
    html+='<pre>'+escape(json.dumps(coverage,indent=2))+'</pre>'
    html+='<p>Upsert verification checks the record count, first/middle/last samples and a vector query. It is not a full-record audit. Numeric presence in retrieval checks is not proof of correct year/row interpretation.</p>'
    html+='<h2>Artifact index</h2><ul>'+''.join('<li><a href="'+escape(path.name)+'">'+escape(path.name)+'</a></li>'
                   for path in sorted(folder.glob('*.json')))+'<li><a href="commands.txt">Remaining commands</a></li></ul></html>'
    if corpus.get("evaluation_set"):
        html=html.replace("Apple financial RAG benchmark",escape(corpus["evaluation_set"]))
    (folder/"report.html").write_text(html,encoding="utf-8")
    print("Published local report: "+str(folder/"report.html"))


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder",type=Path,help="Separate evaluation artifact directory")
    main(parser.parse_args().folder)

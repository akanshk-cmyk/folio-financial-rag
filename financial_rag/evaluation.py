"""Context: save raw answers first, then optionally run paid RAGAS evaluation.

Offline lexical overlap is intentionally not presented as faithfulness/precision/recall.
"""

from importlib.metadata import distributions
import json
import os
import math
from pathlib import Path
import hashlib
import time

from .core import stable_id
from .generation import SYSTEM, load_environment, resolve_model, nebius_base_url
from .chains import build_rag_chain


def run_experiments(fixed, semantic, questions, embedder, output: Path,
                    provider="extractive", model=None, retrieval_backend="local", pinecone_index=None,
                    resume=False):
    """Context: B and C share the exact same candidates; all settings are logged."""
    if not questions or any(not item.get("reference") or "REPLACE WITH" in item["reference"]
                            for item in questions):
        raise ValueError("Supply manually verified reference answers, not the question template.")
    if len({item["id"] for item in questions}) != len(questions):
        raise ValueError("Question IDs must be unique.")
    load_environment()
    if provider in {"groq","openai","nebius"}:
        model = resolve_model(model,provider=provider)
    if fixed.manifest["corpus"] != semantic.manifest["corpus"]:
        raise ValueError("Experiment indexes have different corpora.")
    for key in ("embedding_model", "embedding_revision", "embedding_window_policy"):
        if fixed.manifest[key] != semantic.manifest[key]:
            raise ValueError(f"Experiment indexes differ in {key}.")
    if fixed.manifest.get("chunking_implementation") != semantic.manifest.get("chunking_implementation"):
        raise ValueError("Experiment indexes use different chunking implementations; rebuild both.")
    if not fixed.vectors or not semantic.vectors:
        raise ValueError("Walkthrough experiments require dense indexes, not lexical substitutes.")
    if fixed.manifest["strategy"] != "fixed" or semantic.manifest["strategy"] != "semantic":
        raise ValueError("Supply fixed and semantic indexes in the correct order.")
    if retrieval_backend not in {"local", "pinecone"}:
        raise ValueError("Unknown retrieval backend.")
    searchers = {}
    if retrieval_backend == "pinecone":
        from .cloud import PineconeSearch
        pinecone_index = pinecone_index or os.getenv("PINECONE_INDEX", "financial-rag")
        searchers = {"fixed": PineconeSearch(pinecone_index, "fixed"),
                     "semantic": PineconeSearch(pinecone_index, "semantic")}
    manifest = {"fixed": fixed.manifest, "semantic": semantic.manifest,
                "candidates": 10, "top_n": 3, "context_chars": 24000,
                "provider": provider, "model": model, "temperature": 0,
                "provider_base_url": nebius_base_url() if provider=="nebius" else None,
                "orchestration": "langchain-lcel-v1",
                "retrieval_backend": retrieval_backend, "pinecone_index": pinecone_index,
                "prompt_hash": stable_id(SYSTEM),
                "questions_hash": stable_id(json.dumps(questions, sort_keys=True)),
                "code_fingerprints": {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                      for name in ("generation.py","calculator.py","retrieval.py","cloud.py","chunking.py","embedding.py","chains.py","evaluation.py")},
                "packages": {d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]}}
    rows = []
    if resume and output.exists():
        previous=json.loads(output.read_text(encoding="utf-8"))
        if previous["manifest"]!=manifest:
            raise ValueError("Cannot resume an experiment with changed settings or inputs.")
        rows=previous["rows"]
    def checkpoint():
        """Context: retain every completed question even if a later provider request fails."""
        output.parent.mkdir(parents=True,exist_ok=True)
        temporary=output.with_suffix(".tmp")
        temporary.write_text(json.dumps({"manifest":manifest,"rows":rows},indent=2),encoding="utf-8")
        temporary.replace(output)
    for question in questions:
        pools = {}
        for label, index, rerank in (("A", fixed, False), ("B", semantic, False), ("C", semantic, True)):
            previous=next((row for row in rows if row["question_id"]==question["id"] and row["configuration"]==label),None)
            if previous and previous["status"] not in {"request-failed","skipped-no-baseline"}:
                if label=="B":
                    pools["semantic"]=previous["candidate_pool"]
                continue
            started = time.perf_counter()
            chain = build_rag_chain(index, embedder, question.get("filters", {}),
                                    candidates=10, top_n=3, rerank=rerank, provider=provider, model=model,
                                    saved_candidates=pools.get("semantic") if label == "C" else None,
                                    search_backend=searchers.get(index.manifest["strategy"]))
            if label=="C" and "semantic" not in pools:
                result={"status":"skipped-no-baseline","answer":"","evidence":{},"candidate_pool":[]}
            else:
                try:
                    result = chain.invoke({"question": question["question"]})
                except (RuntimeError,ValueError) as exc:
                    result={"status":"request-failed","answer":"","evidence":{},"candidate_pool":[],
                            "error_type":type(exc).__name__,"error":str(exc)}
            pool = result["candidate_pool"]
            if label == "B" and result["status"]!="request-failed":
                pools["semantic"] = pool
            row={"configuration": label, "question_id": question["id"],
                         "question": question["question"], "reference": question["reference"],
                         "candidate_pool": pool, "seconds": time.perf_counter() - started,
                         "filters": question.get("filters", {}), **result}
            if previous:
                row["previous_infrastructure_failures"]=previous.get("previous_infrastructure_failures",[])+[previous]
                rows[rows.index(previous)]=row
            else:
                rows.append(row)
            checkpoint()
            print(f"{question['id']} / {label}: {result['status']} saved.",flush=True)
    checkpoint()
    return rows


def evaluate_ragas(run_path: Path, output: Path, judge_model=None, provider="nebius"):
    """Context: evaluator versions/models are explicit; actual API calls may exceed metric count."""
    load_environment()
    if provider not in {"groq","openai","nebius"}:
        raise ValueError("Choose nebius, groq or openai for RAGAS judging.")
    key=provider.upper()+"_API_KEY"
    if not os.getenv(key):
        raise ValueError(f"Set {key} in the environment or .env to evaluate answers.")
    judge_model = resolve_model(judge_model, judge=True,provider=provider)
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall
        from ragas.llms import LangchainLLMWrapper
        from ragas.run_config import RunConfig
        if provider in {"openai","nebius"}:
            from langchain_openai import ChatOpenAI
        else:
            from langchain_groq import ChatGroq
    except ImportError as exc:
        raise RuntimeError("Install evaluation extras: uv sync --extra evaluation") from exc
    data = json.loads(run_path.read_text(encoding="utf-8"))
    if any(row["status"] != "generated" for row in data["rows"]):
        raise ValueError("RAGAS requires generated answers; do not judge preview/failed results as answers.")
    samples = [SingleTurnSample(user_input=row["question"], response=row["answer"],
                               retrieved_contexts=list(row["evidence"].values()), reference=row["reference"])
               for row in data["rows"]]
    # Context: these three metrics use only the selected provider's judge, not an
    # embedding service. bypass_n splits multi-completion requests into single
    # prompts; low concurrency reduces bursts without bypassing provider rate limits.
    judge_options = {"reasoning_effort": "low"} if judge_model.startswith("openai/gpt-oss-") else {}
    if provider in {"openai","nebius"}:
        # Context: RAGAS parses JSON metric decisions. Nebius must enforce the
        # same JSON output contract as generation rather than occasionally
        # echoing a reference answer as unstructured text.
        format_options={"model_kwargs":{"response_format":{"type":"json_object"}}} if provider=="nebius" else {}
        judge=ChatOpenAI(model=judge_model,temperature=0,n=1,max_tokens=4096,
                         api_key=os.environ[key],base_url=nebius_base_url() if provider=="nebius" else "https://api.openai.com/v1",
                         **(judge_options if provider=="nebius" else {}), **format_options)
    else:
        judge=ChatGroq(model=judge_model,temperature=0,n=1,max_tokens=4096,**judge_options)
    metrics = [Faithfulness(), LLMContextPrecisionWithReference(), LLMContextRecall()]
    result = evaluate(EvaluationDataset(samples=samples),
                      metrics=metrics,
                      llm=LangchainLLMWrapper(judge, bypass_n=True),
                      run_config=RunConfig(max_workers=1),
                      raise_exceptions=True)
    records = result.to_pandas().to_dict(orient="records")
    for record, row in zip(records, data["rows"]):
        record.update(configuration=row["configuration"], question_id=row["question_id"])
    # Context: aggregate only actual finite evaluator scores; never substitute
    # walkthrough values or turn failed/NaN scores into apparent successes.
    summary = {}
    for label in dict.fromkeys(row["configuration"] for row in records):
        group = [row for row in records if row["configuration"] == label]
        summary[label] = {"questions": len(group)}
        for metric in metrics:
            values = [float(row[metric.name]) for row in group
                      if isinstance(row.get(metric.name), (int, float)) and math.isfinite(row[metric.name])]
            summary[label][metric.name] = sum(values) / len(values) if values else None
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"judge_provider": provider, "judge_model": judge_model, "embedding_model": None,
                                  "judge_base_url": nebius_base_url() if provider=="nebius" else None,
                                  "source_manifest": data["manifest"], "summary": summary, "rows": records},
                                 indent=2, default=str), encoding="utf-8")

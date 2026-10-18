"""Context: build a self-contained, unexecuted Colab notebook from reviewed modules.

Only source code and verified question templates are embedded. No .env, credentials,
PDFs, cached models, database contents, or previous answers enter the notebook.
"""

import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULES = ["__init__.py", "core.py", "embedding.py", "calculator.py", "generation.py",
           "retrieval.py", "chains.py", "evaluation.py", "demo_schemas.py", "pg_store.py",
           "colab_runtime.py", "colab_demo.py", "research_agents.py", "demo_evaluation.py"]


def main():
    cells = []

    def markdown(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip()+"\n"})

    def code(text):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": text.strip()+"\n"})

    markdown("""# Financial Document Intelligence Pipeline RAG
## RAGs to Riches
*Turning boring financial reports into traceable answers.*

This self-contained notebook uses Apple FY2024/FY2025 PDFs, LangChain LCEL,
Groq, Hugging Face BGE, temporary PostgreSQL + pgvector/full-text search,
FlashRank, and four application roles. No Pinecone or Agno key is required.

**Execution status:** every code cell is unexecuted and has no saved outputs.
Installation and database setup are explicit cells. Demo queries and judging are
separate and disabled by default. Turn their toggles on intentionally; external
research and Groq requests use network access and account quotas.

Choose a GPU runtime (T4 if available). CPU also works more slowly. Database,
documents, embeddings, and generated files disappear with the Colab runtime.
Reinitialization preserves existing records; reset is a separate opt-in cell.
""")
    markdown("""## Architecture
```mermaid
flowchart TD
 A[Upload Apple PDFs] --> B[PyPDF layout text and page metadata]
 B --> C[Financial table quality gate]
 C --> D[Fixed 512 tokens / 64 overlap]
 C --> E[Semantic 65th percentile]
 D --> F[Normalized Hugging Face embeddings]
 E --> F
 F --> G[PostgreSQL vector 768 and full-text search]
 G --> H[Metadata-filtered dense and keyword ranks]
 H --> I[RRF then optional FlashRank]
 I --> J[LCEL: exact retained evidence to Groq]
 J --> K[Citation and calculator validation]
 K --> L[AnswerResult and saved evidence/settings]
 L --> M[Deterministic checks, structured judge and RAGAS]
 N[Web search and accessible articles] --> O[Web research role]
 N --> P[Stock research role]
 G --> P
 Q[Timestamped yfinance observations] --> P
 O --> J
 P --> J
```
PostgreSQL keyword ranking uses `ts_rank_cd`, not BM25. Dense search is exact
over prefiltered rows for this small corpus. Web content is excluded from the
controlled four-configuration comparison.
""")
    markdown("""## 1. Installation — core
This cell installs packages only. It does not initialize services, download BGE
weights, or run questions. A working preinstalled PyTorch build is constrained so
the resolver cannot silently replace it. Pip itself does support CUDA builds.
""")
    code('''# Context: install core orchestration/model clients in the notebook's kernel environment.
import json, os, subprocess, sys
from pathlib import Path

def pip_group(packages):
    command = [sys.executable, "-m", "pip", "install", "--quiet"]
    if Path("/content/financial_rag_constraints.txt").exists():
        command += ["--constraint", "/content/financial_rag_constraints.txt"]
    subprocess.run(command + packages, check=True)

# Context: preserve Colab's existing PyTorch/CUDA installation.
# Avoid unnecessary upgrades or reinstalls that could change its compatibility.
probe = subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__)"],
                       capture_output=True, text=True)
ORIGINAL_TORCH = probe.stdout.strip() if probe.returncode == 0 else None
if ORIGINAL_TORCH:
    Path("/content/financial_rag_constraints.txt").write_text(f"torch=={ORIGINAL_TORCH}\\n")
pip_group(["langchain-core==0.3.86", "langchain-text-splitters==0.3.11",
           "langchain-experimental==0.3.4", "langchain-groq==0.3.8",
           "pydantic>=2.7,<3", "groq==0.37.1", "sentence-transformers==5.7.0"])
print("Core packages installed; no model weights loaded yet.")''')
    markdown("## 2. Installation — storage and ingestion\nThese Python drivers are separate from the PostgreSQL system-service cell.")
    code('''# Context: install the database driver, vector adapter, and PDF reader only.
pip_group(["psycopg[binary]>=3.2,<4", "pgvector>=0.3,<0.5", "pypdf>=5,<7"])
print("Storage and ingestion packages installed.")''')
    markdown("## 3. Installation — research and evaluation\nNo web queries, market requests, or judge calls run in this cell.")
    code('''# Context: add explicit research/evaluation dependencies without running their tools.
pip_group(["ddgs>=9,<10", "newspaper4k>=0.9,<1", "yfinance>=0.2,<2",
           "flashrank==0.2.10", "ragas==0.3.9"])
after = subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__)"],
                       check=True, capture_output=True, text=True).stdout.strip()
assert not ORIGINAL_TORCH or after == ORIGINAL_TORCH, "PyTorch changed; verify the CUDA build before continuing."
import torch
print({"torch": after, "cuda_available": torch.cuda.is_available()})''')
    markdown("""## 4. Project source
The following cell contains the reviewed Python modules. Expand it to inspect
context comments and docstrings. It writes source files only and makes no API calls.
""")
    sources = {f"financial_rag/{name}": (ROOT/"financial_rag"/name).read_text(encoding="utf-8") for name in MODULES}
    questions = json.loads((ROOT/"examples/apple_verified_questions.json").read_text(encoding="utf-8"))
    code("# Context: materialize bundled source and verified development questions in the temporary runtime.\n"
         + "MODULE_SOURCE = " + repr(sources) + "\n"
         + "REFERENCE_QUESTIONS = " + repr(questions) + "\n"
         + '''ROOT = Path("/content/financial_rag_demo")
ROOT.mkdir(parents=True, exist_ok=True)
for relative, content in MODULE_SOURCE.items():
    destination = ROOT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
Path("runs").mkdir(exist_ok=True)
print(f"Wrote {len(MODULE_SOURCE)} source modules; credentials and PDFs are not bundled.")''')
    markdown("""## 5. Configuration and Colab Secrets
Add `GROQ_API_KEY` in Colab Secrets and allow notebook access. Optional secrets
`GROQ_MODEL`, `GROQ_JUDGE_MODEL`, and `HF_TOKEN` override defaults. Keys never appear
in saved results. Model access depends on your Groq account.
""")
    code('''# Context: load secrets without displaying them; keep configuration and execution separate.
from google.colab import userdata

def secret(name, default=None):
    try:
        return userdata.get(name) or default
    except Exception:
        return default

key = secret("GROQ_API_KEY")
if not key:
    raise ValueError("Add GROQ_API_KEY to Colab Secrets and enable notebook access.")
os.environ["GROQ_API_KEY"] = key
os.environ["GROQ_MODEL"] = secret("GROQ_MODEL", "openai/gpt-oss-120b")
os.environ["GROQ_JUDGE_MODEL"] = secret("GROQ_JUDGE_MODEL", os.environ["GROQ_MODEL"])
if secret("HF_TOKEN"):
    os.environ["HF_TOKEN"] = secret("HF_TOKEN")
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RETRIEVAL_SETTINGS = {"candidates": 10, "top_n": 5, "context_chars": 24000}
DEFAULT_STRATEGY, DEFAULT_RERANK = "semantic", True
RUN_DEMOS = False
RUN_EVALUATION = False
RUN_COMPARISON = False
print({"generator": os.environ["GROQ_MODEL"], "judge": os.environ["GROQ_JUDGE_MODEL"],
       "embedding_device": EMBEDDING_DEVICE, "api_key_configured": True})''')
    markdown("""## 6. Initialize temporary PostgreSQL + pgvector
This cell explicitly installs PostgreSQL/build tools, starts the local service,
and builds pinned pgvector if needed. It creates a dedicated demo role/database
only when absent and commits `CREATE EXTENSION vector` before driver registration.
Running it again preserves data. It is guarded against execution outside Colab.
""")
    code('''# Context: initialize local runtime storage; never display its generated password or DSN.
from financial_rag.colab_runtime import initialize_postgres
from financial_rag.pg_store import PGStore
database_status = initialize_postgres()
store = PGStore(os.environ["DATABASE_URL"])
store.initialize()
print(database_status)
print(store.counts())''')
    markdown("## 7. Explicit reset — optional\nLeave this disabled to preserve runtime data across initialization.")
    code('''# Context: reset is isolated, explicit, and limited to the demo's tables.
RESET_DEMO_DATA = False
if RESET_DEMO_DATA:
    store.reset(confirmation="DELETE_DEMO_DATA")
    print("Demo records cleared. Re-run ingestion before querying.")
else:
    print("No data deleted.")''')
    markdown("""## 8. Upload and inspect PDFs
Upload `aapl-20240928.pdf` and `aapl-20250927.pdf`. PyPDF retains layout text and
physical PDF page numbers. A statement-row/units quality gate must pass before
indexing. If it fails, inspect/repair extraction instead of indexing damaged tables.
""")
    code('''# Context: copy uploaded PDFs into this runtime without following uploaded path names.
from google.colab import files
uploaded = files.upload()
Path("data/raw").mkdir(parents=True, exist_ok=True)
for filename, content in uploaded.items():
    name = Path(filename).name
    if not name.lower().endswith(".pdf"):
        raise ValueError("Upload PDF files only.")
    (Path("data/raw") / name).write_bytes(content)
print("PDFs saved locally.")''')
    code('''# Context: verify document identity, page provenance, financial row values, and units.
from financial_rag.colab_demo import parse_uploaded_pdf
PDF_PATHS = {2024: "data/raw/aapl-20240928.pdf", 2025: "data/raw/aapl-20250927.pdf"}
SOURCE_URLS = {
  2024: "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm",
  2025: "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm",
}
parsed = [parse_uploaded_pdf(PDF_PATHS[year], year, SOURCE_URLS[year]) for year in (2024, 2025)]
for document in parsed:
    print(document.metadata.title, document.quality_checks)
    statement_page = document.quality_checks[0]["page"]
    print(next(text for page, text in document.pages if page == statement_page)[:4500])''')
    markdown("""## 9. Initialize embeddings and store both chunking strategies
Fixed chunks target **512 tokens with 64-token overlap**, using BGE's tokenizer.
This differs intentionally from the earlier CLI's character baseline. Semantic
chunks use the 65th percentile. Both preserve page provenance. Long semantic
inputs are explicitly windowed; token-length diagnostics are recorded.
""")
    code('''# Context: download one pinned BGE model, select the device, and create the application.
from financial_rag.embedding import Embedder
from financial_rag.colab_demo import FinancialDemo
embedder = Embedder(EMBEDDING_MODEL)
embedder.model.to(EMBEDDING_DEVICE)
demo = FinancialDemo(store, embedder)
print({"model": embedder.name, "revision": embedder.revision, "device": str(embedder.model.device)})''')
    code('''# Context: generate normalized vectors and PostgreSQL full-text representations, preserving both filings.
ingestion_report = demo.ingest(parsed)
Path("runs/ingestion-report.json").write_text(json.dumps(ingestion_report, indent=2), encoding="utf-8")
print(ingestion_report)
print(store.counts())''')
    markdown("""## 10. Define the four roles
These definitions make no requests. Research roles use bounded Groq tool calls;
the document role uses one explicit LCEL evidence path; the evaluation role sees
the exact retained evidence. Calculator requests require cited operands.
""")
    code('''# Context: define reusable roles without executing example questions.
from financial_rag.demo_schemas import QuestionInput, AnswerResult, EvaluationResult
from financial_rag.research_agents import ResearchRoles
from financial_rag.demo_evaluation import judge_answer, write_single_ragas_run
from financial_rag.evaluation import evaluate_ragas
research = ResearchRoles(demo)

def agent_1_web(question):
    """Discover URLs, read accessible articles, then generate a source-grounded answer."""
    return research.run(question, role="web")

def agent_2_financial(question, fiscal_years=None, rerank=DEFAULT_RERANK):
    """Use exactly one retained hybrid evidence path for filing Q&A."""
    return demo.qa(QuestionInput(question=question, fiscal_years=fiscal_years,
        strategy=DEFAULT_STRATEGY, rerank=rerank, **RETRIEVAL_SETTINGS))

def agent_3_stock(question):
    """Combine cited filing evidence, readable web sources, and timestamped market observations."""
    return research.run(question, role="stock")

def agent_4_evaluate(result, reference=None, expected_evidence=None):
    """Separate deterministic evidence/calculation checks from subjective Groq scores."""
    return judge_answer(result, reference, expected_evidence)
print("Four roles defined. No queries executed.")''')
    markdown("""## 11. Prepare verified development questions
The ten reference answers were checked against the supplied income statements.
They are a smoke set, not a diverse held-out benchmark. Add risks, missing evidence,
tables, and calculation questions before drawing broad quality conclusions.
""")
    code('''# Context: attach current document IDs and observed statement pages to reference questions.
by_year = {document.metadata.fiscal_year: document for document in parsed}
questions = []
for item in REFERENCE_QUESTIONS:
    year = item["filters"]["fiscal_year"]
    document = by_year[year]
    questions.append({"id": item["id"], "question": item["question"], "reference": item["reference"],
        "fiscal_years": [year], "expected_evidence": [{"document_id": document.metadata.document_id,
            "pdf_page": document.quality_checks[0]["page"]}]})
Path("runs/questions.json").write_text(json.dumps(questions, indent=2), encoding="utf-8")
print(f"Prepared {len(questions)} reference questions; no answers generated.")''')
    markdown("## 12. Demo query — web research\nEnable `RUN_DEMOS` deliberately. Search snippets discover sources; read articles support claims.")
    code('''# Context: an optional web example, separate from all setup cells.
if RUN_DEMOS:
    web_answer = agent_1_web("What are the applications of artificial intelligence in finance?")
    display(web_answer.model_dump())
else:
    print("Web example disabled; set RUN_DEMOS=True to execute.")''')
    code('''# Context: provide a real article URL if you want this separate article-reading example.
ARTICLE_URL = ""
if RUN_DEMOS and ARTICLE_URL:
    article_answer = agent_1_web(f"Read and summarize this article using only its accessible content: {ARTICLE_URL}")
    display(article_answer.model_dump())
else:
    print("Article example disabled or no URL supplied.")''')
    markdown("## 13. Demo query — financial Q&A")
    code('''# Context: capture the actual generated answer and exact evidence for later evaluation.
if RUN_DEMOS:
    captured_answer = agent_2_financial("What were Apple's total net sales in FY2025?", [2025])
    display(captured_answer.model_dump())
else:
    print("Filing Q&A disabled.")''')
    markdown("## 14. Demo query — cross-year calculation")
    code('''# Context: growth must use source-backed operands and the constrained Decimal calculator.
if RUN_DEMOS:
    growth_answer = agent_2_financial("What was Apple's revenue growth from FY2024 to FY2025?", [2024, 2025])
    display(growth_answer.model_dump())
else:
    print("Cross-year calculation disabled.")''')
    markdown("## 15. Demo query — stock research\nFiscal results and current market observations must remain distinct. Unavailable articles or market fields are reported, not guessed.")
    code('''# Context: this role may make multiple live research and market-data calls.
if RUN_DEMOS:
    stock_answer = agent_3_stock("Analyze Apple using its FY2024 and FY2025 filings, recent news, and available market data.")
    display(stock_answer.model_dump())
else:
    print("Stock research disabled.")''')
    markdown("## 16. Evaluate a captured answer\nNo retrieval is rerun here. Subjective judge scores remain separate from deterministic checks.")
    code('''# Context: judge only an answer that actually exists, using its original evidence.
if RUN_EVALUATION and "captured_answer" in globals():
    reference = next(item for item in questions if item["id"] == "aapl-2025-sales")
    captured_evaluation = agent_4_evaluate(captured_answer, reference["reference"], reference["expected_evidence"])
    display(captured_evaluation)
    write_single_ragas_run(captured_answer, reference["reference"], "runs/captured-ragas-input.json")
    evaluate_ragas(Path("runs/captured-ragas-input.json"), Path("runs/captured-ragas-results.json"))
else:
    print("Evaluation disabled or no captured answer exists.")''')
    markdown("""## 17. Four-configuration comparison
| Configuration | Chunking | Hybrid search | FlashRank |
|---|---|---|---|
| A | Fixed tokens | Enabled | Disabled |
| B | Fixed tokens | Enabled | Enabled |
| C | Semantic | Enabled | Disabled |
| D | Semantic | Enabled | Enabled |

All configurations share documents, encoder revision, generator, prompt, filters,
candidate limits, and context budget. Reranked variants reuse the same candidates.
Web/market research is excluded. Forty generated answers and subsequent judge
calls can consume time and quota. No walkthrough scores are assumed.
""")
    code('''# Context: run all four configurations only when explicitly enabled.
if RUN_COMPARISON:
    comparison_rows = demo.compare(questions, "runs/four-config-answers.json")
    print(f"Captured {len(comparison_rows)} actual answers and their evidence.")
else:
    print("Four-configuration comparison disabled.")''')
    code('''# Context: score saved answers and display measured results, not predetermined winners.
if RUN_EVALUATION and RUN_COMPARISON:
    evaluate_ragas(Path("runs/four-config-answers.json"), Path("runs/four-config-ragas.json"))
    measured = json.loads(Path("runs/four-config-ragas.json").read_text())
    display(measured["summary"])
    print("Review numerical correctness, citation meaning, latency, and failures alongside these scores.")
else:
    print("Comparison evaluation disabled.")''')
    markdown("""## 18. Outputs, limitations, and cleanup
`financial_runs` retains answers with evidence/settings. JSON files in `runs/`
retain ingestion checks, questions, captured answers, and actual judge scores.
Download desired artifacts before deleting the runtime. No automatic reset occurs.

Limitations: text PDFs only; the quality gate checks selected statement rows,
not all disclosures. Semantic chunking does not guarantee intact tables. Windowed
embedding and 512-token reranking have different input coverage. Exact quotations
and arithmetic checks cannot prove correct financial interpretation. Current web
and market data can change; the ten lookup questions are development examples.
The notebook's PostgreSQL service and all live roles need execution in Colab.
""")
    output = ROOT/"notebooks/Financial_Document_Intelligence_Colab.ipynb"
    for number, cell in enumerate(cells):
        cell["id"] = hashlib.sha256(f"{number}:{cell['source']}".encode()).hexdigest()[:12]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"cells": cells, "metadata": {"colab": {"name": output.name},
        "accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}, indent=1), encoding="utf-8")
    print(f"Built {output.name}: {len(cells)} cells, all unexecuted.")


if __name__ == "__main__":
    main()

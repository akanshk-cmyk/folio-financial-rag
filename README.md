# Folio ? Financial Report Intelligence

*Read company reports, compare financial results, and get answers with source evidence.*

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)
![Streamlit app](https://img.shields.io/badge/App-Streamlit-red)
![Evaluation complete](https://img.shields.io/badge/RAGAS-45_responses_evaluated-green)

These badges describe project requirements and measured evaluation status; they are not live CI checks.

## About / Overview

Folio helps users understand annual reports without searching through every page.
Upload a company's 10-K PDF, compare its financial statements, and ask questions.
Answers include the source passages and calculation details so you can check the results.

**How it works:** PDF ? extract text and tables ? split into passages ? search relevant evidence ? generate an answer with Nebius ? check citations and numbers ? show the answer and evidence.

Optional adaptive QA checks evidence relevance, retries retrieval, and checks answer grounding.
The Overview calculates figures directly from recognized statements.
Instructions inside reports are treated as untrusted content.

## Key Features

- Upload text-based PDFs up to 50 MB and save them locally.
- Compare income statements, balance sheets, and cash flows across fiscal years.
- Explore revenue charts, financial ratios, and calculation formulas.
- Ask questions and inspect cited passages, page references, and calculation audits.
- Search locally or use Pinecone; optionally rerank evidence with FlashRank.
- Save answers, export results, and compare retrieval methods using RAGAS.

**Limits:** Scanned PDFs need OCR before upload. Unrecognized layouts or missing figures can leave results unavailable. Answers can report insufficient evidence and still need review.

## Demo / Screenshots

Start the app using the instructions below, then open **http://localhost:8501**.
For saved examples, open the [evaluation scores](docs/evaluation/postfix-summary.json) or [questions, answers, and source evidence](docs/evaluation/answers.html) in a browser. These are HTML reports, not app screenshots.

## Prerequisites & Tech Stack

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/), and internet access for initial model downloads and hosted services.

| Part | Technology |
|---|---|
| Interface | Streamlit |
| PDF extraction | PyMuPDF |
| Retrieval pipeline | LangChain, local search, optional Pinecone |
| Embeddings | HuggingFace BGE, 768 dimensions |
| Evidence reranking | FlashRank |
| Answer generation and evaluation judge | Nebius |
| Financial calculations | Python Decimal |
| Evaluation | RAGAS |

### Install and run

Run these commands from the project folder in PowerShell:

```powershell
uv sync --locked --extra demo --extra pdf --extra models --extra cloud --extra nebius --extra evaluation
# For first-time setup only; keep an existing .env file.
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Add your credentials to .env, then start the app.
uv run --no-sync streamlit run streamlit_app.py
```

### Basic tutorial

1. Open **Document library**, upload a PDF, enter the company and fiscal year, and select **Save and index filing**.
2. In **Workspace settings**, use **Local files** and choose the filing. Add another year's report to compare periods.
3. Open **Company overview** or **Financial statements** and select fiscal years.
4. Open **Ask a question**, choose Nebius, enter a question, and click **Answer**.
5. Review **Supporting evidence**, **View source**, and any calculation audit. Revisit answers in **Saved insights**.

Uploads create local indexes. Pinecone search requires a matching dense snapshot already uploaded to the selected namespace; uploading a PDF does not automatically publish it to Pinecone.

## Configuration / Environment Variables

Set values in your local `.env`; never commit API keys.

| Variable | Purpose |
|---|---|
| `NEBIUS_API_KEY` | Required for Nebius answers and RAGAS judging |
| `NEBIUS_BASE_URL` | `https://api.tokenfactory.nebius.com/v1/` |
| `NEBIUS_MODEL` | Generation model; evaluated with `openai/gpt-oss-120b` |
| `NEBIUS_JUDGE_MODEL` | Evaluation model; evaluated with `openai/gpt-oss-120b` |
| `PINECONE_API_KEY` | Required only for Pinecone access |
| `PINECONE_INDEX` | Pinecone index name; default `financial-rag` |
| `PINECONE_NAMESPACE` | Optional namespace; must match the uploaded snapshot |
| `SEC_USER_AGENT` | Organization/contact identifier when downloading SEC filings |

Overview calculations and local indexing do not need a generation API key.
Nebius uses an OpenAI-compatible client, but requests go to Nebius and use the Nebius key.

## Testing

Run the automated tests:

```powershell
uv run --no-sync python -m unittest discover -s tests -v
```

The last recorded full test run passed **165 tests**. Tests cover retrieval, financial calculations, citations, uploads, and provider integration behavior; live service quality is measured separately.

### Latest RAGAS evaluation

Measured **October 18, 2026**, using Apple FY2024/FY2025 filings and 15 questions per configuration. All **45 responses** were evaluated, with no excluded questions or application validation failures.

| Configuration | Faithfulness | Context precision | Context recall |
|---|---:|---:|---:|
| Fixed chunks | 0.267 | 0.222 | 0.267 |
| Semantic chunks | 0.967 | 0.756 | 0.933 |
| Semantic + FlashRank | 0.767 | 0.667 | 0.733 |

Faithfulness checks answer support, precision checks passage relevance, and recall checks evidence coverage. Higher is better.

There were **28 responses with findings** and **17 insufficient-evidence responses**. Semantic search answered 14 of 15 questions with findings and performed best on this set. All methods left the accounting-policy question unanswered. RAGAS scores are model judgments, not proof of financial accuracy.

The comparison used the same corpus, embeddings, generation settings, and corrected statement tools. Semantic and reranked searches shared candidate pools; adaptive QA was off. All metrics have 15 finite scores per configuration. Earlier partial scores covered 11 questions per configuration.

## Roadmap / Future Improvements

- Add OCR and improve support for complex financial tables.
- Improve evidence coverage for accounting policies and long passages.
- Test more companies, filing layouts, and financial questions.
- Add stronger checks for restatements, reporting periods, and answer completeness.

## Contributing

Use a separate branch and keep changes focused. Describe the problem, your change, and how you tested it. Run the relevant tests before submitting a pull request. Keep credentials and private reports out of contributions; preserve source references when changing retrieval or calculations.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements / Credits

Built with Streamlit, LangChain, HuggingFace/Sentence Transformers, PyMuPDF, Pinecone, FlashRank, Nebius, and RAGAS. Apple SEC filings provide the evaluation evidence.

<img width="1907" height="906" alt="image" src="https://github.com/user-attachments/assets/41e7d8d0-7566-4f3d-bfe9-d9904b8991cd" />
<img width="1582" height="897" alt="image" src="https://github.com/user-attachments/assets/10827636-5dfe-4a84-af61-5eaf4fa26541" />
<img width="1547" height="861" alt="image" src="https://github.com/user-attachments/assets/86f5229c-fc6f-48c7-b5e4-02c8522df432" />
<img width="1560" height="567" alt="image" src="https://github.com/user-attachments/assets/af19b74a-a263-4c4f-8060-3efbe10891e5" />
<img width="1562" height="860" alt="image" src="https://github.com/user-attachments/assets/c5f5da39-10c9-4985-8642-e3b11c0cccf8" />




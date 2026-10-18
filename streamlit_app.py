"""Context: local Streamlit demo. Launch with `uv run streamlit run streamlit_app.py`.

Heavy models and hosted requests execute only after explicit form submissions.
"""

import json
import os
import time
from pathlib import Path
from html import escape

import streamlit as st

from financial_rag.core import Index
from financial_rag.demo_ui import (ROOT, available_indexes, index_uploaded_pdf,
                                   postgres_catalog, run_local_question, save_answer, supporting_evidence)
from financial_rag.generation import load_environment, resolve_model
from financial_rag.overview import render_overview, postgres_overview_chunks
from financial_rag.ui_theme import apply_theme


@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedder(name, revision):
    """Context: reuse model weights across UI reruns, not private document contents."""
    from financial_rag.embedding import Embedder
    return Embedder(name, revision)


def show_result(result):
    """Context: render saved generation and its original evidence without another search."""
    st.caption(f"Status: {result.get('status', 'unknown')} · {result.get('elapsed_seconds', 0):.2f} seconds")
    if result.get("status") == "preview-only":
        st.info("Evidence preview: these are retrieved passages, not an LLM-generated answer.")
        st.write("Inspect the source passages below, or use Answer for a cited response.")
    elif result.get("status") == "validation-failed":
        # Context: a rejected quotation is a generation failure, not evidence that
        # the requested financial data is absent from the retrieved disclosure.
        st.warning("The generated answer could not be verified against its source passages. "
                   "This does not mean the figures are missing from the filing. "
                   "Review the retrieved passages below, or submit the question again.")
        st.caption("Validation detail: "+result.get("validation", "Source verification failed."))
    else:
        st.markdown(result["answer"])
    if result.get("calculations"):
        with st.expander("Calculation audit", expanded=True):
            st.json(result["calculations"])
    if result.get("workflow_trace"):
        with st.expander("Answer workflow"):
            for step in result["workflow_trace"]:
                st.write(step["stage"].capitalize())
                st.json({key:value for key,value in step.items() if key!="stage"})
    cited = supporting_evidence(result)
    if cited:
        st.subheader("Supporting evidence")
        for entry in cited:
            source = entry["source"]
            with st.expander(f"{entry['identifier']} · {source.get('citation', 'Source')}", expanded=True):
                for quote in entry["quotes"]:
                    st.text(quote)
                st.caption(source.get("source", ""))
                if st.button("View source",key="view-source-"+entry["identifier"]):
                    from financial_rag.source_viewer import show_source_dialog
                    show_source_dialog(entry,result.get("evidence",{}).get(entry["identifier"],""))
    elif result.get("evidence"):
        # Context: older previews and refused answers still allow source inspection.
        with st.expander("Retrieved passages"):
            for identifier, passage in result["evidence"].items():
                st.caption(result.get("sources", {}).get(identifier, {}).get("citation", identifier))
                st.text(passage)
    with st.expander("Settings and complete audit"):
        st.json(result)
    st.download_button("Download answer and evidence", json.dumps(result, indent=2, default=str),
                       "financial-rag-answer.json", "application/json")


@st.dialog("Welcome to the Folio demo",width="large")
def show_application_overview():
    """Context: introduce real app capabilities without loading models or calling providers."""
    st.markdown('''<div class="welcome-overview">
        <h1>From number to evidence.</h1>
        <p class="welcome-intro">A financial research workspace built around a simple idea:
        keep the source close to the conclusion.</p>
        <div class="welcome-step"><span>01</span><div><b>Bring your reports. Compare the reporting years.</b>
        <p>Upload a company’s text-based 10-K in Document library. Choose your filing and fiscal years
        in Workspace settings, then explore the overview, balance sheet, income statement, and cash flow.</p></div></div>
        <div class="welcome-step"><span>02</span><div><b>Follow the evidence.</b>
        <p>Use Ask a question to generate a grounded answer with supporting evidence.
        Review the cited excerpts and calculation audit below the answer.</p></div></div>
        <div class="welcome-step"><span>03</span><div><b>Make the workspace yours.</b>
        <p>Revisit previous answers in Saved insights. Export the overview or download an answer
        with its evidence. Uploaded PDFs and search indexes stay saved on this computer.</p></div></div>
        <div class="welcome-footer">The financial overview works without a generation API key.
        Nebius or Groq generation requires your configured account and runs only after submission.
        Unrecognized financial figures remain unavailable.</div></div>''',unsafe_allow_html=True)
    if st.button("Explore the workspace",type="primary"):
        st.rerun()


def main():
    """Context: reruns update controls; only submitted forms mutate data or call models."""
    st.set_page_config(page_title="Financial Report Intelligence", page_icon="📊", layout="wide")
    apply_theme()
    load_environment(ROOT / ".env")
    breadcrumb=st.empty()
    if notice:=st.session_state.pop("upload_notice",None):
        st.success(notice)

    with st.sidebar:
        st.markdown('<div class="workspace-brand"><span class="folio-mark"><i></i><i></i><i></i></span>folio<span>.</span></div>'
                    '<div class="workspace-label">Research workspace</div>', unsafe_allow_html=True)
        pages=["Company overview","Financial statements","Document library","Saved insights","Ask a question","How it works"]
        if target:=st.session_state.pop("pending_page",None):
            st.session_state.workspace_page=target if target in pages else pages[0]
        # Context: existing browser sessions may still point to a removed page.
        if st.session_state.get("workspace_page") not in pages:
            st.session_state.workspace_page=pages[0]
        with st.container(key="workspace-navigation"):
            page=st.radio("Workspace navigation",pages,key="workspace_page",label_visibility="collapsed")
        company_card=st.empty()
        about_requested=st.button("Application overview",help="Open the welcome guide")
        settings=st.expander("Workspace settings",expanded=False)
    with settings:
        backend = st.selectbox("Storage", ["Local files", "PostgreSQL + pgvector"])
        generation_provider = st.radio("Generation provider", ["Nebius", "Groq"])
        provider=generation_provider.lower()
        api_key_name=provider.upper()+"_API_KEY"
        st.caption(f"Answer retrieves supporting evidence and uses {api_key_name} from your environment or .env.")
        model = st.text_input("Generation model", value=resolve_model(provider=provider))
        rerank = st.checkbox("Rerank with FlashRank", value=False,
                              help="The first request may download the local reranker model.")
        top_n = st.slider("Evidence passages", 1, 5, 3)

    index = None
    if backend == "Local files":
        paths = available_indexes()
        choices = [str(path.relative_to(ROOT)) for path in paths]
        if choices:
            # Context: activate a successfully persisted upload on the next rerun.
            pending=st.session_state.pop("pending_filing_index",None)
            if pending in choices:
                st.session_state.filing_index=pending
            if st.session_state.get("filing_index") not in choices:
                st.session_state.filing_index=choices[0]
            choice = settings.selectbox("Filing index", choices,key="filing_index")
            path = ROOT / choice
            signature = (str(path), path.stat().st_mtime_ns)
            if st.session_state.get("index_signature") != signature:
                st.session_state.disk_index = Index.load(path)
                st.session_state.index_signature = signature
            index = st.session_state.disk_index
        else:
            st.info("Upload a filing in Document library to begin.")
        rows = [{"Company": chunk.ticker, "Fiscal year": chunk.fiscal_year,
                 "Document": chunk.doc_id} for chunk in index.chunks] if index else []
        if index:
            settings.caption("Local dense + BM25 retrieval" if index.vectors else
                       "Local BM25 evidence demo · no embedding download needed")
    else:
        st.caption("Connect to an existing PostgreSQL database using DATABASE_URL in .env.")
        if st.button("Connect to PostgreSQL"):
            try:
                from financial_rag.pg_store import PGStore
                if not os.getenv("DATABASE_URL"):
                    raise ValueError("Set DATABASE_URL in .env first.")
                store = PGStore(os.environ["DATABASE_URL"])
                docs, datasets = postgres_catalog(store)
                st.session_state.pg_catalog = (docs, datasets)
            except Exception:
                st.error("Could not read the database. Check DATABASE_URL, pgvector, and the initialized demo tables.")
        docs, datasets = st.session_state.get("pg_catalog", ([], []))
        rows = [{"Company": doc["company"], "Fiscal year": doc["fiscal_year"],
                 "Document": doc["document_id"]} for doc in docs]
        dataset = settings.selectbox("Retrieval dataset", datasets,
            format_func=lambda item: f"{item['strategy']} · {item['chunks']} chunks · {item['dataset_id'][:8]}") if datasets else None

    # Context: fiscal year is an explicit filter rather than an unreliable guess from prose.
    companies = sorted({row["Company"] for row in rows})
    ticker = settings.selectbox("Company", companies) if companies else None
    years_available = sorted({row["Fiscal year"] for row in rows if row["Company"] == ticker})
    years = settings.multiselect("Filing fiscal years", years_available, default=years_available)
    metadata=index.manifest.get("upload_documents",{}) if index else {}
    company_name=next((item["company_name"] for item in metadata.values() if item.get("ticker")==ticker),
                      "Apple Inc." if ticker=="AAPL" else ticker or "Your company")
    company_card.markdown('<div class="company-card"><span class="company-avatar">'+escape(company_name[:1])+
        '</span><div><b>'+escape(company_name)+'</b><small>'+escape(ticker or "No filing selected")+
        ' · Financial disclosures</small></div></div>',unsafe_allow_html=True)
    breadcrumb.markdown('<div class="breadcrumb"><span>Workspace &nbsp; › &nbsp; Companies &nbsp; › &nbsp; '
        '<strong>'+escape(company_name)+'</strong></span><span class="data-status">● &nbsp; Filing data</span></div>',unsafe_allow_html=True)
    st.sidebar.markdown('<div class="workspace-note"><b>A space to explore</b><br>'
                        'Research your selected financial disclosures. Overview figures come from '
                        'the loaded filings.</div>', unsafe_allow_html=True)

    if page in {"Company overview","Financial statements"}:
        if backend=="Local files":
            render_overview(index.chunks if index else [],ticker,years,
                            index.manifest.get("upload_documents",{}) if index else {},
                            statements_only=page=="Financial statements")
        elif rows and dataset and years:
            try:
                from financial_rag.pg_store import PGStore
                overview_chunks=postgres_overview_chunks(PGStore(os.environ["DATABASE_URL"]),dataset["dataset_id"],ticker,years)
                render_overview(overview_chunks,ticker,years,statements_only=page=="Financial statements")
            except Exception:
                st.warning("The selected database filing could not be loaded for the overview.")
        else:
            st.info("Connect to PostgreSQL and select filing years to view the financial overview.")

    if page=="Ask a question":
        st.markdown('<div class="eyebrow">Company intelligence / Research assistant</div>',unsafe_allow_html=True)
        st.title("Ask the filings.")
        st.caption("Grounded answers, with the evidence and calculations behind them.")
        if rows:
            a, b, c = st.columns(3)
            a.metric("Filings", len({row["Document"] for row in rows}))
            b.metric("Fiscal years", len(years_available))
            c.metric("Evidence limit", top_n)
        evidence_source=st.selectbox("Evidence source",["Selected local collection","Pinecone"],disabled=backend!="Local files")
        query_index=index
        if evidence_source=="Pinecone" and backend=="Local files":
            # Context: benchmark search can use its exact published snapshot while
            # overview figures keep the original page-aware filing collection.
            snapshots=[]
            if index and index.vectors:
                snapshots.append(path)
            for candidate in (ROOT/"runs/apple-pinecone-benchmark-v1/semantic.json",
                              ROOT/"runs/apple-pinecone-benchmark-v1/fixed.json"):
                receipt=candidate.with_name("pinecone-"+candidate.stem+".json")
                try:
                    verified=json.loads(receipt.read_text(encoding="utf-8")).get("verification")=="passed"
                except (OSError,ValueError):
                    verified=False
                if verified and candidate.exists() and index and {chunk.ticker for chunk in index.chunks}=={"AAPL"}:
                    snapshots.append(candidate)
            if snapshots:
                snapshot=st.selectbox("Pinecone vector snapshot",snapshots,
                                     format_func=lambda item:str(item.relative_to(ROOT)))
                query_index=Index.load(snapshot)
            pinecone_index=st.text_input("Pinecone index",value=os.getenv("PINECONE_INDEX","financial-rag"))
            default_namespace=query_index.manifest.get("strategy","") if snapshots and snapshot.parent.name=="apple-pinecone-benchmark-v1" else ""
            pinecone_namespace=st.text_input("Pinecone namespace",value=os.getenv("PINECONE_NAMESPACE",default_namespace),
                help="Use the namespace uploaded from this exact dense snapshot. Fixed/semantic identify chunking strategies, not question topics.")
            if st.button("Connect to Pinecone"):
                try:
                    from financial_rag.cloud import inspect_service
                    connection=inspect_service(pinecone_index)
                    if connection.get("exists") and connection.get("ready"):
                        st.success("Connected to "+pinecone_index)
                        st.json(connection)
                    else:
                        st.warning("The selected Pinecone index is absent or not ready.")
                except Exception as exc:
                    st.error("Pinecone connection failed: "+type(exc).__name__+". Check your API key and network access.")
        with st.form("question_form"):
            if pending_question:=st.session_state.pop("pending_question",None):
                st.session_state.research_question=pending_question
            elif "research_question" not in st.session_state:
                st.session_state.research_question=f"What were {ticker or 'the company'}'s total revenues in fiscal year {max(years) if years else 2025}?"
            question = st.text_area("Your question",key="research_question")
            adaptive=st.checkbox("Adaptive evidence checks",value=backend=="Local files",disabled=backend!="Local files",
                help="Grade evidence, retry a broader search within your selected company/years, and check the answer. Adds LLM requests. Available for local collections and Pinecone.")
            # Context: one submission retrieves evidence and generates a cited answer.
            with st.container(key="question-actions"):
                generate_answer=st.form_submit_button("Answer",disabled=not rows,type="primary")
        if generate_answer:
            if not question.strip() or not years:
                st.warning("Enter a question and select at least one filing year.")
            elif not os.getenv(api_key_name):
                st.error(f"Set {api_key_name} in .env or the environment first.")
            else:
                try:
                    started = time.perf_counter()
                    with st.spinner("Generating answer and supporting evidence…"):
                        if backend == "Local files":
                            search_backend=None
                            if evidence_source=="Pinecone":
                                from financial_rag.cloud import PineconeSearch
                                if not query_index.vectors:
                                    raise ValueError("Pinecone requires a dense snapshot uploaded with its matching namespace. Select that Filing index first.")
                                search_backend=PineconeSearch(pinecone_index,pinecone_namespace)
                            embedder = load_embedder(query_index.manifest["embedding_model"], query_index.manifest["embedding_revision"]) if query_index.vectors else None
                            progress=st.empty()
                            result = run_local_question(query_index, question, ticker, years,
                                provider, rerank, 10, top_n, embedder, model,adaptive=adaptive,
                                search_backend=search_backend,
                                namespace=pinecone_namespace if evidence_source=="Pinecone" else None,
                                on_event=lambda event:progress.caption("Answer workflow: "+event["stage"]))
                            progress.empty()
                        else:
                            from financial_rag.colab_demo import FinancialDemo
                            from financial_rag.demo_schemas import DocumentMetadata, QuestionInput
                            from financial_rag.pg_store import PGStore
                            settings = dataset["settings"]
                            embedder = load_embedder(settings["embedding_model"], settings["revision"])
                            demo = FinancialDemo(PGStore(os.environ["DATABASE_URL"]), embedder, model, provider)
                            demo.documents = {doc["document_id"]: DocumentMetadata(**doc) for doc in docs}
                            demo.datasets = {dataset["strategy"]: dataset["dataset_id"]}
                            captured = demo.qa(QuestionInput(question=question, company=ticker, fiscal_years=years,
                                strategy=dataset["strategy"], rerank=rerank, top_n=max(top_n,len(years))))
                            result = {**captured.audit["generation"], "audit": captured.model_dump(mode="json")}
                        result["elapsed_seconds"] = time.perf_counter() - started
                        path = save_answer(result)
                        st.session_state.last_result = result
                        st.session_state.last_result_path = str(path.relative_to(ROOT))
                except (ValueError, RuntimeError, ImportError) as exc:
                    st.error(str(exc))
                except Exception:
                    st.error("The request failed. Check the configured database, model availability, and installed optional dependencies.")
        if "last_result" in st.session_state:
            st.caption(f"Last submitted question: {st.session_state.last_result['question']}")
            st.caption(f"Saved to {st.session_state.last_result_path}; reflects settings at submission.")
            show_result(st.session_state.last_result)

    if page=="Document library":
        st.markdown('<div class="eyebrow">Research workspace / Document library</div>',unsafe_allow_html=True)
        st.title("Your source of truth.")
        if rows:
            st.dataframe([dict(items) for items in sorted({tuple(row.items()) for row in rows})], hide_index=True)
        st.subheader("Upload a company’s 10-K")
        st.caption("Upload a text-based PDF, up to 50 MB. The PDF, parsed pages, and search index are saved on this computer. Scanned PDFs require OCR before upload.")
        if backend!="Local files":
            st.info("Uploads create a local index. Switch Storage to Local files to search the saved report.")
        with st.form("upload_form"):
            upload = st.file_uploader("Financial filing PDF", type=["pdf"])
            company_name=st.text_input("Company name",placeholder="Example: Microsoft Corporation")
            upload_ticker = st.text_input("Filing ticker",placeholder="Example: MSFT")
            upload_year = st.number_input("Filing fiscal year", min_value=1990, max_value=2100, value=2025)
            currency=st.selectbox("Reporting currency (verify in the filing)",["USD","EUR","GBP","CAD","JPY","CNY","AUD","CHF","INR"])
            source = st.text_input("Original source URL (optional)")
            upload_search=st.selectbox("Upload search method",["BM25 — quick local search","Hugging Face + BM25 — hybrid search"],
                help="Hybrid search downloads BAAI/bge-base-en-v1.5 on first use and embeds the filing locally. No generation API key is required for indexing.")
            merge=st.checkbox("Add to the selected local collection",value=False,
                help="Use matching search settings to combine filings for cross-year questions. Existing saved collections are preserved.")
            add = st.form_submit_button("Save and index filing",type="primary")
        if add:
            if upload is None:
                st.warning("Choose a PDF first.")
            else:
                try:
                    with st.spinner("Parsing PDF and saving the searchable filing…"):
                        if merge and (backend!="Local files" or index is None):
                            raise ValueError("Select an existing local collection before adding to it.")
                        embedder=None
                        if upload_search.startswith("Hugging Face"):
                            from financial_rag.embedding import MODEL
                            revision=index.manifest.get("embedding_revision") if merge and index else None
                            embedder=load_embedder(MODEL,revision)
                        fresh = index_uploaded_pdf(upload.getvalue(), upload_ticker, int(upload_year), source,
                            index if merge else None,company_name=company_name,currency=currency,embedder=embedder,
                            storage_dir=ROOT/"data/uploads")
                        saved_path=Path(fresh.manifest["saved_index"])
                        st.session_state.pending_filing_index=str(saved_path.relative_to(ROOT))
                        st.session_state.pending_page="Company overview"
                        st.session_state.pop("last_result",None)
                        st.session_state.upload_notice=f"Saved {len(fresh.chunks)} searchable passages for {upload_ticker.upper()} FY{int(upload_year)}."
                    st.rerun()
                except (ValueError, RuntimeError, ImportError) as exc:
                    st.error(str(exc))
                except Exception:
                    st.error("The PDF could not be indexed. Check that it is readable, unlocked, and contains extractable text.")

    if page=="Saved insights":
        # Context: saved results are local audit records; viewing them never regenerates an answer.
        st.markdown('<div class="eyebrow">Research workspace / Saved insights</div>',unsafe_allow_html=True)
        st.title("Research, remembered.")
        saved=[]
        for saved_path in sorted((ROOT/"runs/streamlit").glob("answer-*.json"),reverse=True):
            try:
                result=json.loads(saved_path.read_text(encoding="utf-8"))
                if result.get("filters",{}).get("ticker")==ticker:
                    saved.append((saved_path,result))
            except (OSError,ValueError):
                continue
        if saved:
            selected=st.selectbox("Saved question",range(len(saved)),format_func=lambda value:saved[value][1].get("question",saved[value][0].name))
            show_result(saved[selected][1])
        else:
            st.info("No saved answers for this company yet. Submit a question in Ask a question to create one.")

    if page=="How it works":
        st.title("From disclosure to insight.")
        st.write("PDF → page-aware parsing → chunks → filtered retrieval → optional FlashRank → selected LLM → citation and calculation checks.")
        st.write("Uploads save the original PDF, parsed pages and a local index under data/uploads. Choose BM25 or Hugging Face + BM25; optional FlashRank applies when a question is submitted. "
                 "PostgreSQL uses pgvector + full-text search + RRF, matching the Colab retrieval path.")
        st.write("Grounded answers and their exact evidence are saved in runs/streamlit. "
                 "Preview passages are labeled explicitly and are not generated financial conclusions.")
        st.write("The Colab notebook remains the place to run the controlled four-configuration comparison and RAGAS evaluation.")
        st.caption("Hosted generation requires explicit submission. The app does not automatically initialize PostgreSQL, publish vectors or run evaluations.")

    # Context: show once per browser session; the sidebar button can reopen the guide at any time.
    if about_requested or not st.session_state.get("welcome_seen",False):
        st.session_state.welcome_seen=True
        show_application_overview()


if __name__ == "__main__":
    main()

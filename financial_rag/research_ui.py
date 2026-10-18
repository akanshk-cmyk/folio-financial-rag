"""Context: local research-agent UI; external requests require an explicit submitted action."""

import json
import os
from pathlib import Path


def render_research_agent(index,ticker,years,provider,model,rerank,top_n,load_embedder,root):
    """Context: review source permissions, then collect evidence or run the tool-calling planner."""
    import streamlit as st
    from .evidence_agent import EvidencePolicy, EvidenceResearchAgent

    st.markdown('<div class="eyebrow">Research workspace / Tool-calling research agent</div>',unsafe_allow_html=True)
    st.title("Evidence before conclusions.")
    st.caption("Gather filing passages, official disclosures and timestamped Yahoo Finance observations. Review every tool call and source before generating a report.")
    if index is None:
        st.info("Choose Local files and a filing index to use the research agent. Upload a company’s PDF in Document library if needed.")
        return
    document_only=st.checkbox("Document-only evaluation — disable all external sources",value=True,
        help="Only the selected local index, company and fiscal years can supply evidence. Web/Yahoo tools are not registered in this mode.")
    permissions=st.columns(2)
    with permissions[0]:
        official=st.checkbox("Permit official filing / investor-relations search",value=False,disabled=document_only)
    with permissions[1]:
        yahoo=st.checkbox("Permit Yahoo Finance market observations",value=False,disabled=document_only)
    domains=st.text_input("Approved investor-relations domains",placeholder="Example: investor.apple.com, investor.nvidia.com",
        disabled=document_only or not official,help="SEC is included. Add only company-controlled official domains you have verified. The agent cannot expand this list.")
    if not document_only and official and not os.getenv("SEC_USER_AGENT","").strip():
        st.caption("SEC downloads require SEC_USER_AGENT with your organization and real contact email in .env. Uploaded filings and Yahoo do not require it.")
    with st.form("research-agent-form"):
        question=st.text_area("Research question",f"Analyze {ticker}'s financial performance for fiscal year {max(years) if years else 2025}, identifying supported findings and evidence limitations.")
        official_urls=st.text_area("Official document URLs (optional; one per line)",
            help="Direct evidence collection reads these approved-domain URLs instead of searching. The selected latest fiscal year must match the disclosure header.",
            disabled=document_only or not official)
        write_report=st.checkbox("Generate a cited financial research report after tool collection",value=False,
            help="Uses the selected generation provider and your account quota. Missing sections remain unavailable; forecasts or expert quotes are never invented.")
        col1,col2=st.columns(2)
        with col1:
            collect=st.form_submit_button("Collect evidence — direct tools",type="primary",disabled=not ticker or not years)
        with col2:
            run_agent=st.form_submit_button("Run tool-calling agent",type="primary",disabled=not ticker or not years)
    st.caption(f"Company: {ticker or 'not selected'} · Fiscal years: {', '.join(map(str,years)) or 'not selected'} · Planner/report provider: {provider}. Direct evidence collection without a report makes no LLM request.")
    if collect or run_agent:
        try:
            if (run_agent or write_report) and not os.getenv(provider.upper()+"_API_KEY"):
                raise ValueError("Set "+provider.upper()+"_API_KEY in .env, or collect evidence without generating a report.")
            approved=("sec.gov",)+tuple(domain.strip().lower() for domain in domains.split(",") if domain.strip() and not document_only and official)
            policy=EvidencePolicy(ticker=ticker,fiscal_years=tuple(years),document_only=document_only,
                allow_official=official and not document_only,allow_yahoo=yahoo and not document_only,
                official_domains=tuple(dict.fromkeys(approved)))
            with st.status("Collecting and recording source evidence…",expanded=True) as progress:
                def tool_event(event):
                    """Context: expose actual dispatch events while research runs, not fabricated progress."""
                    if event["stage"]=="started":
                        progress.write("Calling "+event["tool"])
                        progress.json(event["arguments"])
                    elif event.get("error"):
                        progress.write(event["tool"]+": "+event["error"])
                    else:
                        progress.write(event["tool"]+f" completed · {event['source_count']} evidence passages retained")
                embedder=load_embedder(index.manifest["embedding_model"],index.manifest["embedding_revision"]) if index.vectors else None
                agent=EvidenceResearchAgent(index,policy,embedder=embedder,rerank=rerank,top_n=top_n,
                    provider=provider,model=model,artifact_dir=Path(root)/"runs/research-agent",on_tool_event=tool_event)
                result=agent.run(question,use_planner=run_agent,generate_report=write_report,
                    official_urls=tuple(url.strip() for url in official_urls.splitlines() if url.strip()))
                st.session_state.last_research_result=result
                progress.update(label="Research saved: "+result["status"],state="error" if result["status"] in {"planner-failed","report-failed","validation-failed"} else "complete")
        except (ValueError,RuntimeError,ImportError) as exc:
            st.error(str(exc))
        except Exception:
            st.error("Research could not complete. Check installed research dependencies, provider access and source availability.")
    result=st.session_state.get("last_research_result")
    if not result:
        return
    st.divider()
    st.caption(f"Saved research: {result['question']} · Status: {result['status']} · Settings below reflect this saved run.")
    if result.get("planner_error"):
        st.warning(result["planner_error"]["message"])
    if result.get("generation_error"):
        st.warning(result["generation_error"]["message"])
    if result.get("report"):
        st.markdown(result["report"])
        st.download_button("Download cited research report",result["report"],"financial-research-report.md","text/markdown")
    if not result["sources"]:
        st.info("No supported evidence was collected. Inspect Tool-call history for source failures or scope restrictions.")
    a,b,c=st.columns(3)
    a.metric("Evidence passages",len(result["sources"]))
    b.metric("Tool calls",len(result["tool_log"]))
    c.metric("External sources used","Yes" if result["external_data_used"] else "No")
    st.subheader("Source evidence bundle")
    for source in result["sources"]:
        with st.expander(source["source_id"]+" · "+source["source_type"]+" · "+source["location"]):
            st.caption(source["citation"])
            st.caption("Source: "+source["document_url"])
            st.text(source["text"])
            st.markdown("**Source metadata**")
            st.json({key:value for key,value in source.items() if key not in {"text","tables"}})
    with st.expander("Tool-call history"):
        st.json(result["tool_log"])
    with st.expander("Financial analysis handoff"):
        st.json(result.get("financial_analysis") or {"status":"No recognized financial statement analysis available."})
    with st.expander("Source policy and evaluation controls"):
        st.json({"policy":result["source_policy"],"comparison_eligible":result["comparison_eligible"],
                 "planner_used":result["planner_used"],"provider":result["provider"],"model":result["model"],
                 "saved_path":result["saved_path"]})
    st.download_button("Download complete evidence bundle",json.dumps(result,indent=2,default=str),
        "research-evidence-bundle.json","application/json")

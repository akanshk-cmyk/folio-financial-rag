"""Context: display original filing evidence, with highlights that never rewrite text."""

from html import escape
import re
from urllib.parse import urlparse


def source_details(source):
    """Context: use explicit metadata; support older saved answers with citation-only metadata."""
    citation=source.get("citation","")
    body=citation.strip("[]")
    doc_id=source.get("document_id") or body.split(",",1)[0]
    year=source.get("fiscal_year")
    if not year:
        match=re.search(r"(?:^|_)((?:19|20)\d{2})(?:_|$)",doc_id)
        year=match[1] if match else None
    location=source.get("location") or "Source location unavailable"
    section=source.get("section_path") or "Financial disclosure"
    # Context: commas in section titles must not be mistaken for field separators.
    match=re.search(r", ((?:PDF )?Page .*|Anchor .*|Block .*|Chunk .*?)\]$",citation)
    if match and not source.get("location"):
        location=match[1]
        if not source.get("section_path"):
            section=citation[len(doc_id)+3:match.start()]
    doc_type=source.get("doc_type")
    if not doc_type:
        match=re.search(r"(?:^|_)(10-K|10-Q|Transcript)(?:_|$)",doc_id,re.I)
        doc_type=match[1].upper() if match else "financial disclosure"
    title=f"FY {year} annual report" if year and doc_type=="10-K" else (
        f"FY {year} {doc_type}" if year else "Financial disclosure")
    return {"title":title,"document_id":doc_id,"section":section,"location":location,
            "synthetic":bool(source.get("synthetic")),"citation":citation,"url":source.get("source","")}


def excerpt_html(passage, quotes):
    """Context: retain exact source characters and escape HTML before marking literal spans.

    Surround each cited span with nearby complete lines. Disjoint excerpts are
    separated visibly; ellipses are never inserted inside a quoted source span.
    """
    intervals=[]
    for quote in quotes:
        if not isinstance(quote,str) or not quote.strip():
            continue
        start=passage.find(quote)
        if start<0:
            continue
        end=start+len(quote)
        left=passage.rfind("\n",0,max(0,start-160))+1
        right=passage.find("\n",min(len(passage),end+160))
        intervals.append((left,len(passage) if right<0 else right,start,end))
    if not intervals:
        return "",False
    windows=[]
    for left,right,_,_ in sorted(intervals):
        if windows and left<=windows[-1][1]:
            windows[-1]=(windows[-1][0],max(right,windows[-1][1]))
        else:
            windows.append((left,right))
    blocks=[];tabular=False
    for left,right in windows:
        spans=[]
        for _,_,start,end in sorted(intervals,key=lambda interval:interval[2]):
            if start>=left and end<=right:
                if spans and start<=spans[-1][1]:
                    spans[-1]=(spans[-1][0],max(end,spans[-1][1]))
                else:
                    spans.append((start,end))
        rendered=[];cursor=left
        for start,end in spans:
            rendered.extend((escape(passage[cursor:start]),"<mark>"+escape(passage[start:end])+"</mark>"))
            cursor=end
        rendered.append(escape(passage[cursor:right]))
        tabular=tabular or bool(re.search(r"\d[\d,.]*[ \t]{2,}\S",passage[left:right]))
        blocks.append("".join(rendered))
    return '<div class="source-gap">Separate source excerpt</div>'.join(
        '<div class="source-excerpt-text">'+block+'</div>' for block in blocks),tabular


def source_card_html(source, passage, quotes):
    """Context: build the screenshot-style source card using verified excerpts only."""
    details=source_details(source)
    excerpt,tabular=excerpt_html(passage,quotes)
    label="SYNTHETIC TEST SOURCE / NOT AN ACTUAL FILING" if details["synthetic"] else "SOURCE EVIDENCE / ORIGINAL DISCLOSURE EXCERPT"
    return ('<article class="source-viewer"><div class="source-eyebrow">'+label+'</div>'
            '<h1>'+escape(details["title"])+'</h1><div class="source-badge">'
            +escape(details["location"])+" &nbsp; / &nbsp; "+escape(details["section"])+"</div>"
            +'<div class="source-excerpt'+(' source-tabular' if tabular else '')+'">'
            +(excerpt or "No verified excerpt is available.")+'</div><div class="source-footer">'
            +'Highlighted text is quoted from the saved evidence. Surrounding text preserves its original wording. '
            +'Calculated figures are supported by cited inputs; they may not appear as reported figures in the filing.'
            +'</div></article>')


def show_source_dialog(entry, passage):
    """Context: opening evidence reads saved text only; no retrieval or LLM requests occur."""
    import streamlit as st
    @st.dialog("Source evidence",width="large")
    def display():
        source=entry["source"]
        st.markdown(source_card_html(source,passage,entry["quotes"]),unsafe_allow_html=True)
        st.caption(source.get("citation",""))
        url=source.get("source","")
        parsed=urlparse(url)
        if parsed.scheme in {"https","http"} and parsed.hostname and not parsed.username and not parsed.password:
            st.link_button("Open original filing",url)
        with st.expander("Complete retrieved passage"):
            st.text(passage)
        st.download_button("Download source excerpt","\n\n".join(entry["quotes"]),
                           "financial-source-excerpt.txt","text/plain",key="source-excerpt-download")
    display()

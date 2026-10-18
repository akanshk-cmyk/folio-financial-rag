"""Context: source-backed company overview without hosted generation or model downloads.

Read numeric rows with their actual year headers from selected filing chunks.
No reference answers or hardcoded financial figures supply dashboard values.
Unsupported layouts produce missing metrics rather than guessed columns.
"""

from decimal import Decimal
import re

from .calculator import calculate
from .financial_analysis import analyze_financial_statements, render_financial_analysis
from .display_format import amount, financial_value, panel_heading, percentage
from .comparisons import combine_reports, period_value, year_growth


METRICS = {
    "sales":("Total net sales","Net sales","USD millions"),
    "cost_sales":("Total cost of sales","Cost of sales / revenue","USD millions"),
    "gross_profit":("Gross margin","Gross profit","USD millions"),
    "rd":("Research and development","Research and development","USD millions"),
    "sga":("Selling, general and administrative","Selling, general and administrative","USD millions"),
    "opex":("Total operating expenses","Total operating expenses","USD millions"),
    "operating_income":("Operating income","Operating income","USD millions"),
    "other_income":("Other income/(expense), net","Other income / (expense), net","USD millions"),
    "pretax_income":("Income before provision for income taxes","Income before income taxes","USD millions"),
    "income_tax":("Provision for income taxes","Income tax expense","USD millions"),
    "net_income":("Net income","Net income","USD millions"),
    "basic_eps":("Basic","Basic EPS","USD per share"),
    "eps":("Diluted","Diluted EPS","USD per share"),
}
PRODUCTS=("iPhone","Mac","iPad","Wearables, Home and Accessories","Services")
ALIASES={
    "sales":("Total net sales","Total revenues","Total revenue","Net revenues","Net sales","Revenues","Revenue"),
    "net_income":("Net income","Net earnings","Net income (loss)","Net earnings (loss)"),
    "operating_income":("Operating income","Income from operations","Operating income (loss)","Operating profit"),
    "eps":("Diluted","Diluted earnings per share","Diluted net income per share","Diluted earnings (loss) per share"),
    "rd":("Research and development","Research and development expenses"),
    "sga":("Selling, general and administrative","Selling, general and administrative expenses","Sales, general and administrative"),
    "opex":("Total operating expenses","Operating expenses"),
    "gross_profit":("Gross profit","Gross margin"),
    "cost_sales":("Total cost of sales","Cost of revenue","Total cost of revenue","Cost of revenues","Cost of sales","Cost of goods sold"),
    "other_income":("Other income/(expense), net","Total other income, net","Other income (expense), net","Other income, net"),
    "pretax_income":("Income before provision for income taxes","Income before income tax","Income before income taxes","Earnings before income taxes"),
    "income_tax":("Provision for income taxes","Income tax expense","Income taxes","Provision for income tax"),
    "basic_eps":("Basic","Basic earnings per share","Basic net income per share"),
}
STATEMENT=re.compile(r"^[ \t]*consolidated[ \t]+statements?[ \t]+of[ \t]+(?:operations|income|earnings|loss)"
                     r"(?:[ \t]*\(continued\))?[ \t]*$",re.I|re.M)
# Context: accept explicit calendar-date column labels without borrowing fiscal years from metadata.
MONTH=r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
DATE=MONTH+r"[ \t]+(?:[1-9]|[12]\d|3[01]),?[ \t]+(?:19|20)\d{2}"
YEARS=r"(?:19|20)\d{2}(?:[ \t]+(?:19|20)\d{2})+"
HEADER=re.compile(r"^[ \t]*(?:"+YEARS+r"|"+DATE+r"(?:[ \t]+"+DATE+r")+)[ \t]*$",re.M|re.I)
BALANCE_HEADING=re.compile(r"^[ \t]*consolidated[ \t]+(?:balance[ \t]+sheets?|statements?[ \t]+of[ \t]+financial[ \t]+position)(?:[ \t]*\(continued\))?[ \t]*$",re.I|re.M)
CASH_HEADING=re.compile(r"^[ \t]*consolidated[ \t]+statements?[ \t]+of[ \t]+cash[ \t]+flows?(?:[ \t]*\(continued\))?[ \t]*$",re.I|re.M)
# Context: aliases identify disclosed rows, not calculated or inferred totals.
BALANCE_ROWS={
    "cash":("Cash and cash equivalents",("Cash and cash equivalents",)),
    "receivables":("Accounts receivable, net",("Accounts receivable, net","Accounts receivable")),
    "inventories":("Inventories",("Inventories","Inventory")),
    "current_assets":("Total current assets",("Total current assets",)),
    "ppe":("Property and equipment, net",("Property, plant and equipment, net","Property and equipment, net")),
    "goodwill":("Goodwill",("Goodwill",)),
    "total_assets":("Total assets",("Total assets",)),
    "payables":("Accounts payable",("Accounts payable",)),
    "current_liabilities":("Total current liabilities",("Total current liabilities",)),
    "long_term_debt":("Long-term debt",("Long-term debt","Long-term debt, net")),
    "total_liabilities":("Total liabilities",("Total liabilities",)),
    "retained_earnings":("Retained earnings / accumulated deficit",("Retained earnings","Accumulated deficit","Retained earnings (accumulated deficit)")),
    "equity":("Total shareholders’ equity",("Total shareholders’ equity","Total shareholders' equity","Total stockholders’ equity","Total stockholders' equity","Total equity")),
    "liabilities_equity":("Total liabilities and equity",("Total liabilities and shareholders’ equity","Total liabilities and shareholders' equity","Total liabilities and stockholders’ equity","Total liabilities and stockholders' equity","Total liabilities and equity")),
}
CASH_ROWS={
    "net_income":("Net income",("Net income","Net earnings")),
    "depreciation":("Depreciation and amortization",("Depreciation and amortization",)),
    "stock_compensation":("Share-based compensation",("Share-based compensation expense","Stock-based compensation expense","Stock-based compensation")),
    "operating_cash":("Net cash from operating activities",("Net cash provided by operating activities","Cash generated by operating activities","Net cash provided by (used in) operating activities","Net cash used in operating activities")),
    "capital_expenditure":("Property and equipment purchases",("Payments for acquisition of property, plant and equipment","Purchases of property and equipment","Purchases related to property and equipment and intangible assets","Additions to property and equipment")),
    "investing_cash":("Net cash from investing activities",("Net cash used in investing activities","Net cash provided by investing activities","Cash generated by investing activities","Cash used in investing activities","Net cash provided by (used in) investing activities","Cash generated by/(used in) investing activities")),
    "dividends":("Dividends paid",("Dividends paid","Payments for dividends and dividend equivalents","Payments of dividends")),
    "financing_cash":("Net cash from financing activities",("Net cash used in financing activities","Cash used in financing activities","Net cash provided by financing activities","Cash generated by financing activities","Net cash provided by (used in) financing activities")),
    "cash_change":("Change in cash and cash equivalents",("Change in cash and cash equivalents","Net increase (decrease) in cash and cash equivalents")),
    "cash_beginning":("Cash and cash equivalents, beginning",("Cash and cash equivalents at beginning of period","Cash and cash equivalents, beginning of year")),
    "cash_ending":("Cash and cash equivalents, ending",("Cash and cash equivalents at end of period","Cash and cash equivalents, end of year","Cash, cash equivalents, and restricted cash and cash equivalents, ending balances")),
}


def extract_statement_rows(scope,heading,definitions,year,currency_for,period_type):
    """Context: search only the selected document's matching statement pages."""
    pages=[chunk for chunk in scope if heading.search(chunk.text.split("Supplementary detected table;")[0])]
    records={}
    for key,(title,aliases) in definitions.items():
        record=next((record for alias in aliases for chunk in pages
                     if (record:=read_row(chunk,alias,year,"USD millions",currency_for(chunk)))),None)
        if record:
            record.update(title=title,period_type=period_type)
            records[key]=add_change(record)
    return records


def read_row(chunk,label,year,unit,currency="USD",component=None):
    """Context: use explicit table columns; reject shifted or unrecognized layouts."""
    primary=chunk.text.split("Supplementary detected table;")[0]
    pattern=r"^[ \t]*"+re.escape(label)+r"[ \t]+(?P<values>[$€£¥]?[ \t]*\(?-?\d[^\n]*)$"
    for match in re.finditer(pattern,primary,re.M|re.I):
        if component:
            # Context: repeated debt rows must be tied to their original current
            # or non-current section, not selected by occurrence order alone.
            headings=list(re.finditer(r"^[ \t]*((?:Non-current|Current)[ \t]+liabilities)[ \t]*:.*$",
                                      primary[:match.start()],re.I|re.M))
            if not headings or headings[-1][1].casefold()!=component.casefold():
                continue
        headers=list(HEADER.finditer(primary[:match.start()]))
        if not headers:
            continue
        header=headers[-1]
        previous_line=primary[:header.start()].splitlines()[-1:] or [""]
        dated_start=header.start()-len(previous_line[0])-1 if re.search(MONTH,previous_line[0],re.I) else header.start()
        years=[int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b",header.group())]
        if len(set(years))!=len(years) or years[0]!=chunk.fiscal_year or year not in years:
            continue
        if unit!="%" and "%" in match["values"]:
            continue
        if label.lower() in {"diluted","basic"}:
            # Context: repeated Diluted rows can refer to EPS or share counts; never conflate them.
            subsections=list(re.finditer(r"^.*(?:per.share|shares|earnings per|income per).*$",
                                         primary[header.end():match.start()],re.I|re.M))
            if subsections and re.search(r"weighted|shares (?:used|outstanding)",subsections[-1].group(),re.I):
                continue
        values=re.findall(r"\(?-?\d[\d,]*(?:\.\d+)?\)?",match["values"])
        # Context: count columns before treating a parenthesized leading numeral as a footnote.
        # Otherwise a cash outflow such as (974) can be silently discarded as a footnote.
        if len(values)==len(years)+1 and re.match(r"^\(\d{1,2}\)[ \t]+",match["values"]):
            values=values[1:]
        if len(values)!=len(years):
            continue
        parsed={period:Decimal(value.replace(",","").replace("(","-").replace(")",""))
                for period,value in zip(years,values)}
        # Context: retain the disclosed scale; never silently label thousands as millions.
        actual_unit=unit
        if unit!="%":
            if not currency:
                continue
            prefix=primary[:header.start()]
            # Context: share-count exceptions cannot override the financial table's monetary scale.
            monetary_prefix=re.sub(r"\bexcept\b[^\n]*","",prefix,flags=re.I)
            scales=list(re.finditer(r"\bin\s+(millions|thousands|billions)\b",monetary_prefix,re.I))
            if not scales:
                continue
            declared=set(re.findall(r"\b(?:USD|EUR|GBP|CAD|JPY|CNY|AUD|CHF|INR)\b",prefix))
            if declared and declared!={currency}:
                continue
            if ("€" in primary and currency!="EUR") or ("£" in primary and currency!="GBP"):
                continue
            if ("$" in primary and currency not in {"USD","CAD","AUD"}) or ("¥" in primary and currency not in {"JPY","CNY"}):
                continue
            actual_unit=f"{currency} per share" if unit.endswith("per share") else f"{currency} {scales[-1][1].lower()}"
        return {"label":label,"value":str(parsed[year]),"prior_value":str(parsed[year-1]) if year-1 in parsed else None,
            "year":year,"unit":actual_unit,"currency":currency,"scope":f"{chunk.ticker} consolidated",
            "document_id":chunk.doc_id,"evidence_id":chunk.id,
            "citation":chunk.citation(),"location":chunk.location,"source":chunk.source,
            "quote":primary[header.start():match.end()],"row_quote":match.group(),
            "period_quote":primary[max(0,dated_start):header.end()],
            "component_quote":primary[headings[-1].start():match.end()] if component else match.group()}
    return None


def add_change(record):
    """Context: all financial percentage arithmetic uses the existing Decimal calculator."""
    if record is None or record["prior_value"] is None or Decimal(record["prior_value"])<=0:
        return record
    common={"currency":record.get("currency","USD"),"scale":"per share" if record["unit"].endswith("per share") else record["unit"].split()[-1],
            "basis":"as reported","scope":record.get("scope","consolidated"),"period_type":record.get("period_type","annual"),
            "metric":record["label"],"evidence_id":record["evidence_id"],"quote":record["quote"]}
    inputs=[{**common,"period":f"FY{record['year']}","value":record["value"]},
            {**common,"period":f"FY{record['year']-1}","value":record["prior_value"]}]
    record["change"]=calculate("growth",inputs,{record["evidence_id"]:record["quote"]})
    return record


def financial_overview(chunks,ticker,years,document_metadata=None):
    """Context: each year's overview stays inside one selected source document."""
    document_metadata=document_metadata or {}
    selected=[chunk for chunk in chunks if chunk.ticker==ticker and chunk.fiscal_year in years and not chunk.synthetic]
    reports={}
    for year in sorted(set(years)):
        candidates=[chunk for chunk in selected if chunk.fiscal_year==year
                    and STATEMENT.search(chunk.text)]
        def currency_for(chunk):
            configured=document_metadata.get(chunk.doc_id,{}).get("currency")
            explicit=set(re.findall(r"\b(?:USD|EUR|GBP|CAD|JPY|CNY|AUD|CHF|INR)\b",chunk.text))
            return configured or (next(iter(explicit)) if len(explicit)==1 else "USD" if ticker=="AAPL" else None)
        documents={chunk.doc_id for chunk in candidates
                   if any(read_row(chunk,label,year,METRICS[key][2],currency_for(chunk))
                          for key,labels in ALIASES.items() for label in labels)}
        if len(documents)>1:
            raise ValueError(f"More than one FY{year} report version is selected. Choose an index with one version per filing year.")
        if not documents:
            continue
        document=next(iter(documents))
        statement=[chunk for chunk in candidates if chunk.doc_id==document]
        scope=[chunk for chunk in selected if chunk.doc_id==document]
        metrics={}
        for key,(label,title,unit) in METRICS.items():
            record=next((record for alias in ALIASES[key] for chunk in statement
                         if (record:=read_row(chunk,alias,year,unit,currency_for(chunk)))),None)
            if record:
                record["title"]=title
                metrics[key]=add_change(record)
        gross=next((record for chunk in scope if (record:=read_row(chunk,"Total gross margin percentage",year,"%"))),None)
        if gross:
            gross["title"]="Total gross margin"
            metrics["gross_margin"]=gross
        products={}
        for label in PRODUCTS if ticker=="AAPL" else ():
            # Product rows must come from the disaggregated net-sales table.
            for chunk in scope:
                if not re.search(r"net sales disaggregated|disaggregated net sales",chunk.text,re.I):
                    continue
                record=read_row(chunk,label,year,"USD millions",currency_for(chunk))
                if record:
                    record["title"]=label
                    products[label]=add_change(record)
                    break
        metadata=document_metadata.get(document,{})
        reports[year]={"document_id":document,"ticker":ticker,
            "company_name":metadata.get("company_name") or ("Apple Inc." if ticker=="AAPL" else ticker),
            "currency":next(iter(metrics.values()))["currency"],
            "currency_basis":metadata.get("currency_basis","disclosed currency / existing Apple index"),
            "metrics":metrics,"products":products,
            "balance_sheet":extract_statement_rows(scope,BALANCE_HEADING,BALANCE_ROWS,year,currency_for,"point-in-time"),
            "cash_flow":extract_statement_rows(scope,CASH_HEADING,CASH_ROWS,year,currency_for,"annual"),
            "missing_metrics":[title for key,(_,title,_) in METRICS.items() if key not in metrics]}
        reports[year]["analysis"]=analyze_financial_statements(reports[year],{chunk.id:chunk.text for chunk in scope})
    return reports


def postgres_overview_chunks(store,dataset_id,ticker,years):
    """Context: read only the selected PostgreSQL dataset and metadata scope."""
    from .pg_store import rows_to_hits
    with store.connection() as connection:
        rows=connection.execute("SELECT c.chunk_id,c.document_id,c.pdf_page,c.text,d.company,d.fiscal_year,"
            "d.title,d.source_url,1.0 AS score FROM financial_chunks c "
            "JOIN financial_documents d USING(document_id) "
            "WHERE c.dataset_id=%s AND d.company=%s AND d.fiscal_year=ANY(%s) ORDER BY c.chunk_id",
            [dataset_id,ticker,years]).fetchall()
    return [hit.chunk for hit in rows_to_hits(rows)]


def accounting_amount(value,per_share=False):
    """Context: parentheses are presentation only; source records retain signed values."""
    return amount(value,2 if per_share else None)


def render_statement_table(metrics,year,income_statement=False,comparison_years=None,margin_metrics=None):
    """Context: format verified records as a readable table; source evidence stays in exports."""
    from html import escape
    import streamlit as st
    periods=comparison_years or [year,year-1]
    headers=''.join(f'<th>FY {period}</th>' for period in periods)
    growth_periods=[period for period in periods if period-1 in periods]
    change_headers=''.join(f'<th>YoY FY {period}</th>' for period in growth_periods)

    # Context: income rows follow the statement's financial progression, not dashboard-card order.
    ordered={key:metrics[key] for key in METRICS if key in metrics} if income_statement else metrics
    if income_statement:
        ordered.update({key:record for key,record in metrics.items() if key not in ordered})
    rows=[]
    section=None
    calculations={metric["key"]:metric for metric in (margin_metrics or [])}
    margin_parents={"gross_profit":("gross_margin","Gross margin %"),
                    "operating_income":("operating_margin","Operating margin %"),
                    "net_income":("net_margin","Net profit margin %")}
    displayed_margins=set()

    def margin_row(key,title):
        """Context: use disclosed percentages first; fallback only to the audited calculated margin."""
        source=metrics.get(key)
        calculated=calculations.get(key)
        if source is None and calculated is None:
            return ""
        cells=[]
        for period in periods:
            value=period_value(source,period,year) if source else None
            basis="Reported in the filing"
            if value is None:
                audit=calculated.get("periods",{}).get(str(period),{}) if calculated else {}
                value=audit.get("display_output") if audit.get("status")=="verified" else None
                basis="Calculated: profit / revenue × 100" if value is not None else "No verified input available"
            cells.append('<td title="'+escape(basis)+'">'+percentage(value)+'</td>')
        displayed_margins.add(key)
        return '<tr class="statement-margin"><td>'+escape(title)+'</td>'+''.join(cells)+'<td>—</td>'*len(growth_periods)+'</tr>'
    for key,record in ordered.items():
        if record["unit"]=="%":
            continue
        if income_statement:
            group="Operating expenses" if key in {"rd","sga","opex"} else "Earnings per share" if key in {"basic_eps","eps"} else None
            if group and group!=section:
                rows.append(f'<tr class="statement-section"><td colspan="{1+len(periods)+len(growth_periods)}">{escape(group)}</td></tr>')
            section=group
        cells=''.join('<td>'+accounting_amount(period_value(record,period,year),record["unit"].endswith("per share"))+'</td>' for period in periods)
        changes=[]
        for period in growth_periods:
            change=record["growth_by_year"].get(period) if "growth_by_year" in record else year_growth(record,period,year)
            delta=percentage(change["display_output"],signed=True) if change else "—"
            changes.append('<td>'+escape(delta)+'</td>')
        row_class=' class="statement-total"' if key in {"sales","gross_profit","opex","operating_income","pretax_income","net_income","total_assets","total_liabilities","equity","liabilities_equity","operating_cash","investing_cash","financing_cash"} else ' class="statement-detail"' if income_statement and key in {"rd","sga","basic_eps","eps"} else ''
        rows.append(f'<tr{row_class}><td>{escape(record["title"])}</td>'
                    +cells+''.join(changes)+'</tr>')
        if income_statement and key in margin_parents:
            margin_key,title=margin_parents[key]
            rows.append(margin_row(margin_key,title))
    # Context: a disclosed margin remains visible even when its related amount was not extracted.
    for key,record in metrics.items():
        if record["unit"]=="%" and key not in displayed_margins:
            rows.append(margin_row(key,"Gross margin %" if key=="gross_margin" else record["title"]))
    st.markdown('<div class="statement-scroll">'
                '<table class="statement-table"><thead><tr><th>Line item</th>'
                +headers+change_headers+'</tr></thead><tbody>'
                +''.join(rows)+'</tbody></table></div>',unsafe_allow_html=True)
    share_units=list(dict.fromkeys(record["unit"] for record in metrics.values() if record["unit"].endswith("per share")))
    if share_units:
        st.caption("EPS: "+"; ".join(share_units)+".")
    if displayed_margins:
        st.caption("Margin % rows use disclosed percentages where available; otherwise, verified profit ÷ revenue calculations. Hover over a percentage to see its basis.")


def render_statements(report,year):
    """Context: native tabs switch between three source-backed financial statements."""
    from html import escape
    import streamlit as st
    labels=("Income statement","Balance sheet","Cash flow")
    groups=(report["metrics"],report.get("balance_sheet",{}),report.get("cash_flow",{}))
    # Context: the heading reports actual disclosed monetary units across the statement tabs.
    units=list(dict.fromkeys(record["unit"] for records in groups for record in records.values()
                             if record["unit"]!="%" and not record["unit"].endswith("per share")))
    reporting_units=" / ".join(units) or report.get("currency","Unavailable")
    st.markdown(panel_heading("Behind the performance",reporting_units),unsafe_allow_html=True)
    for tab,label,records in zip(st.tabs(labels),labels,groups):
        with tab:
            if records:
                render_statement_table(records,year,income_statement=label=="Income statement",comparison_years=report.get("comparison_years"),
                                       margin_metrics=report.get("analysis",{}).get("metrics",[]) if label=="Income statement" else None)
            else:
                st.info(f"{label}: Information not available in provided financial disclosures. No rows could be verified in the recognized table layout.")


def render_overview(chunks,ticker,years,document_metadata=None,statements_only=False):
    """Context: reference-inspired dashboard using only extracted, selected filing data."""
    import json
    from html import escape
    import pandas as pd
    import streamlit as st

    section="Financial statements" if statements_only else "Financial overview"
    st.markdown(f'<div class="eyebrow">Company intelligence / {section}</div>', unsafe_allow_html=True)
    try:
        reports=financial_overview(chunks,ticker,years,document_metadata)
    except ValueError as exc:
        st.warning(str(exc))
        return
    if not reports:
        st.info("Overview figures are unavailable: the filing needs a recognized consolidated income statement, explicit units, and readable year columns. You can still search it in Ask a question.")
        return
    heading, controls=st.columns([3,2],vertical_alignment="center")
    with heading:
        st.title(reports[max(reports)]["company_name"])
        st.caption(f"{ticker} · Annual financial disclosures")
    with controls:
        year_column, export_column=st.columns(2,vertical_alignment="bottom")
        with year_column:
            # Context: retain multi-year selection inside a compact dropdown-style popover.
            for period in reports:
                st.session_state.setdefault(f"overview-year-{reports[period]['document_id']}",period==max(reports))
            checked=[period for period in sorted(reports,reverse=True)
                     if st.session_state[f"overview-year-{reports[period]['document_id']}"]]
            label=", ".join(f"FY {period}" for period in checked) or "Select fiscal years"
            with st.container(key="fiscal-year-picker"):
                with st.popover(label,icon=":material/calendar_month:",use_container_width=True):
                    st.caption("Select fiscal years")
                    selected_years=[period for period in sorted(reports,reverse=True)
                        if st.checkbox(f"FY {period}",key=f"overview-year-{reports[period]['document_id']}")]
        with export_column:
            # Context: the exported report retains full evidence, hidden from the overview UI.
            if selected_years:
                exported=combine_reports(reports,selected_years)
                suffix="-".join(str(period) for period in selected_years)
                st.download_button("Export report",json.dumps(exported,indent=2),
                                   f"{ticker.lower()}-fy{suffix}-overview.json","application/json",type="primary")
    if not selected_years:
        st.info("Select at least one fiscal year to view the report.")
        return
    render_selected_report(exported,max(selected_years),statements_only)


def render_selected_report(report,year,statements_only=False):
    """Context: isolate each selected fiscal year's widgets, data, and calculation audit."""
    from html import escape
    import pandas as pd
    import streamlit as st

    metrics=report["metrics"]
    comparison_years=report.get("comparison_years",[year,year-1])
    if len(comparison_years)>2:
        st.markdown('<div class="comparison-periods">'+" · ".join(f"FY {period}" for period in comparison_years)+'</div>',unsafe_allow_html=True)
        st.caption("All comparison years appear together. Overlapping years use the newest selected filing.")
    if report.get("comparison_conflicts"):
        st.info("Some overlapping figures differ between filings. This comparison uses the newest selected filing’s values; Export report records both versions.")
    if report["missing_metrics"]:
        with st.expander("Extraction notes"):
            st.caption("Unavailable in the recognized table layout: "+", ".join(report["missing_metrics"])+".")
    if statements_only:
        with st.container(border=True,key=f"statement-detail-{year}"):
            render_statements(report,year)
        return
    st.divider()
    headline_keys=("sales","net_income","operating_income","eps")
    if len(comparison_years)>2:
        # Context: headline figures use the same combined year columns as the detailed statements.
        headline={key:metrics[key] for key in headline_keys if key in metrics}
        units=list(dict.fromkeys(record["unit"] for record in headline.values() if not record["unit"].endswith("per share")))
        with st.container(border=True):
            st.markdown(panel_heading("Key financial figures"," / ".join(units)),unsafe_allow_html=True)
            render_statement_table(headline,year,comparison_years=comparison_years)
    else:
        for column,key in zip(st.columns(4),headline_keys):
            record=metrics.get(key)
            with column:
                if record is None:
                    st.metric(METRICS[key][1],"Unavailable")
                    continue
                display=financial_value(record["value"],record["unit"])
                change=record.get("change")
                delta=percentage(change["display_output"],signed=True)+f" vs FY{year-1}" if change else None
                st.markdown('<div class="metric-reporting-unit reporting-unit">'+escape(record["unit"])+"</div>",unsafe_allow_html=True)
                st.metric(record["title"],display,delta)

    st.divider()
    main_column, brief_column=st.columns([2.7,1],gap="medium")
    with main_column:
        with st.container(border=True,key=f"overview-chart-{year}"):
            chart_records=report["products"] or {record["title"]:record for key,record in metrics.items()
                                                  if key in {"sales","net_income","operating_income"}}
            chart_units={record["unit"] for record in chart_records.values()}
            chart_unit=next(iter(chart_units)) if len(chart_units)==1 else None
            st.markdown(panel_heading("Revenue across the business." if report["products"] else "Annual financial performance.",chart_unit or "Unavailable"),unsafe_allow_html=True)
            st.caption("Annual comparison" if chart_unit else "Annual comparison unavailable: inconsistent units")
            # Context: annual comparisons use actual prior-year table columns, never invented quarters.
            product_rows=[]
            for label,record in chart_records.items() if chart_unit else ():
                for period in reversed(comparison_years):
                    value=period_value(record,period,year)
                    if value is not None:
                        product_rows.append({"Category":label,"Period":f"FY {period}",
                                             "Amount":float(value)})
            if product_rows:
                periods=[f"FY {period}" for period in reversed(comparison_years)]
                st.vega_lite_chart(pd.DataFrame(product_rows),{
                    "height":255,"mark":{"type":"bar","cornerRadiusTopLeft":2,"cornerRadiusTopRight":2,"size":30},
                    "encoding":{
                        "x":{"field":"Category","type":"nominal","sort":list(chart_records),
                             "axis":{"title":None,"labelAngle":0,"labelLimit":125}},
                        "xOffset":{"field":"Period","sort":periods},
                        "y":{"field":"Amount","type":"quantitative","axis":{"title":None}},
                        "color":{"field":"Period","type":"nominal",
                                 "scale":{"domain":periods,"range":["#e4d6dd","#bb98aa","#583044"][-len(periods):] if len(periods)<=3 else ["#eadfe5","#d4bdc9","#b48da0","#8b5e75","#583044"]},
                                 "legend":{"orient":"top","direction":"horizontal","title":None,"symbolType":"square"}},
                        "tooltip":[{"field":"Category"},{"field":"Period"},
                                   {"field":"Amount","format":",.0f"}]},
                    "config":{"background":"#fcfbf9","view":{"stroke":None},
                              "axis":{"gridColor":"#e8dfe3","gridDash":[2,4],"domain":False,
                                      "labelColor":"#92838a","tickColor":"#e8dfe3"}}
                },use_container_width=True,theme=None)
            else:
                st.info("Chart figures are unavailable in the selected filing layout.")
            sales=metrics.get("sales")
            if sales and sales.get("change"):
                st.caption(f"FY {year} net sales changed "+percentage(sales["change"]["display_output"],signed=True)+f" versus FY {year-1}.")

        # Context: stack chart and ratios in one column so the taller research brief
        # cannot leave an empty row beneath the chart or change panel widths.
        render_financial_analysis(report["analysis"])

    with brief_column:
        # Context: reuse source-backed figures; the reference's fictional subscription and cash claims are not copied.
        highlights=[("Growth","Revenue, in focus.",metrics.get("sales")),
                    ("Profitability","The earnings picture.",metrics.get("operating_income") or metrics.get("net_income")),
                    ("Cash generation","Cash from operations.",report.get("cash_flow",{}).get("operating_cash"))]
        with st.container(key=f"research-brief-panel-{year}"):
            st.markdown('<div class="research-brief-header"><div class="brief-label">'
                        '<span>✧ &nbsp; The research brief</span><span>PREVIEW</span></div>'
                        '<h2>Beyond the numbers.</h2><p class="intro">Three things worth a closer look.<br>'
                        'Every insight has a place to start.</p></div>',unsafe_allow_html=True)
            for number,(topic,title,record) in enumerate(highlights,1):
                text="Information not available in provided financial disclosures."
                if record and len(comparison_years)>2:
                    figures=[f"FY {period}: {financial_value(period_value(record,period,year),record['unit'],include_unit=True)}"
                             for period in comparison_years]
                    text=record["title"]+". "+"; ".join(figures)+"."
                elif record and record.get("value") is not None:
                    text=f"{record['title']}: {financial_value(record['value'],record['unit'],include_unit=True)} in FY{year}."
                    if record.get("change"):
                        text+=" The year-over-year change was "+percentage(record["change"]["display_output"],signed=True)+"."
                st.markdown(f'<div class="brief-item"><div class="eyebrow">{number:02d} / {escape(topic)}</div>'
                            f'<h4>{escape(title)}</h4><p>{escape(text)}</p></div>',unsafe_allow_html=True)
            # Context: the footer transfers a question for review; it never submits a hosted request automatically.
            with st.form(f"brief-question-form-{year}"):
                prompt,send=st.columns([5,1],vertical_alignment="bottom")
                with prompt:
                    brief_question=st.text_input("Ask about this company",placeholder="Ask about this company...",label_visibility="collapsed")
                with send:
                    open_question=st.form_submit_button("→",help="Open your question in the research assistant",type="primary")
                if open_question:
                    if brief_question.strip():
                        st.session_state.pending_question=brief_question.strip()
                        st.session_state.pending_page="Ask a question"
                        st.rerun()
                    else:
                        st.warning("Enter a question first.")
            st.caption("Opens the question page for review. No automatic API request.")


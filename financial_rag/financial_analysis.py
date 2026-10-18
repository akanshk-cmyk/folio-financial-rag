"""Context: audited statement analysis from extracted filing rows, without LLM arithmetic."""

from decimal import Decimal, localcontext, ROUND_HALF_EVEN

from .calculator import canonical_quote, numeric_literals
from .display_format import financial_value


DEFINITIONS=(
    ("gross_margin","Profitability","Gross margin","gross_profit","sales","percent","Gross profit / revenue × 100"),
    ("operating_margin","Profitability","Operating margin","operating_income","sales","percent","Operating income / revenue × 100"),
    ("net_margin","Profitability","Net profit margin","net_income","sales","percent","Net income / revenue × 100"),
    ("current_ratio","Liquidity","Current ratio","current_assets","current_liabilities","ratio","Current assets / current liabilities"),
    ("working_capital","Liquidity","Working capital","current_assets","current_liabilities","difference","Current assets − current liabilities"),
    ("liabilities_assets","Capital structure","Liabilities / assets","total_liabilities","total_assets","percent","Total liabilities / total assets × 100"),
    ("cash_conversion","Cash generation","Operating cash / net income","operating_cash","net_income","ratio","Operating cash flow / net income"),
    ("free_cash_flow","Cash generation","Free cash flow (calculated)","operating_cash","capital_expenditure","free_cash","Operating cash flow − absolute property/equipment purchases"),
    ("fcf_margin","Profitability","Free cash flow margin","operating_cash","sales","free_cash_percent","(Operating cash flow − absolute property/equipment purchases) / revenue × 100"),
    ("equity_assets","Financial position","Equity / assets","equity","total_assets","percent","Total equity / total assets × 100"),
    ("liabilities_equity","Financial position","Liabilities / equity","total_liabilities","equity","ratio","Total liabilities / total equity"),
    ("cash_assets","Financial position","Cash / assets","cash","total_assets","percent","Cash and cash equivalents / total assets × 100"),
    ("cash_liabilities","Financial position","Cash / liabilities","cash","total_liabilities","ratio","Cash and cash equivalents / total liabilities"),
    ("fcf_earnings","Cash quality","Free cash flow / earnings","operating_cash","net_income","free_cash_ratio","(Operating cash flow − absolute property/equipment purchases) / net income"),
    ("capital_intensity","Cash quality","Capital intensity","capital_expenditure","sales","absolute_percent","Absolute property/equipment purchases / revenue × 100"),
    ("operating_cash_margin","Cash quality","Operating cash margin","operating_cash","sales","percent","Operating cash flow / revenue × 100"),
)


def analyze_financial_statements(report,evidence):
    """Context: validate actual source spans, comparability, and accounting identities before ratios."""
    records={**report["metrics"],**report.get("balance_sheet",{}),
             **{key:value for key,value in report.get("cash_flow",{}).items() if key!="net_income"}}
    checks=[]
    invalid={}
    year=next(iter(report["metrics"].values()))["year"]
    period_years=(year,year-1)

    def observation(key,period):
        record=records.get(key)
        field="value" if period==year else "prior_value"
        if record is None or record.get(field) is None:
            raise ValueError("Required filing input is unavailable.")
        if record["document_id"]!=report["document_id"] or record["year"]!=year:
            raise ValueError("Input belongs to a different filing or fiscal period.")
        passage=evidence.get(record["evidence_id"],"")
        quote=canonical_quote(record["quote"],passage)
        value=Decimal(record[field])
        if not value.is_finite() or value not in numeric_literals(quote):
            raise ValueError("Input value is not verified in its source quotation.")
        return {"key":key,"value":str(value),"period":period,"unit":record["unit"],
                "currency":record["currency"],"scope":record["scope"],
                "period_type":record.get("period_type","annual"),"document_id":record["document_id"],
                "evidence_id":record["evidence_id"],"citation":record["citation"],"quote":quote}

    def compatible(inputs,match_period_type=True):
        fields=("unit","currency","scope","document_id","period")
        if match_period_type:
            fields+=("period_type",)
        if any(len({item[field] for item in inputs})!=1 for field in fields):
            raise ValueError("Inputs have incompatible units, currencies, scopes, or periods.")

    identities=(
        ("Gross profit reconciliation",("sales","cost_sales","gross_profit"),"subtract"),
        ("Operating profit reconciliation",("gross_profit","opex","operating_income"),"subtract"),
        ("Net income reconciliation",("pretax_income","income_tax","net_income"),"subtract"),
        ("Balance sheet reconciliation",("total_liabilities","equity","total_assets"),"add"),
    )
    with localcontext() as context:
        context.prec=40
        for period in period_years:
            invalid.setdefault(period,set())
            for label,keys,operation in identities:
                check={"name":label,"period":period,"status":"unavailable"}
                try:
                    inputs=[observation(key,period) for key in keys]
                    compatible(inputs)
                    a,b,total=(Decimal(item["value"]) for item in inputs)
                    calculated=a+b if operation=="add" else a-b
                    check.update(status="passed" if calculated==total else "mismatch",inputs=inputs,
                                 calculated=str(calculated),reported=str(total),difference=str(calculated-total))
                    if calculated!=total:
                        invalid[period].update(keys)
                except ValueError as exc:
                    check["reason"]=str(exc)
                checks.append(check)

        metrics=[]
        for key,category,title,numerator,denominator,kind,formula in DEFINITIONS:
            metric={"key":key,"category":category,"title":title,"formula":formula,"kind":kind,"periods":{}}
            for period in period_years:
                result={"status":"unavailable"}
                try:
                    inputs=[observation(numerator,period),observation(denominator,period)]
                    if kind in {"free_cash_percent","free_cash_ratio"}:
                        inputs.append(observation("capital_expenditure",period))
                    compatible(inputs)
                    if {item["key"] for item in inputs}&invalid[period]:
                        raise ValueError("A required input failed an accounting reconciliation.")
                    a,b=(Decimal(item["value"]) for item in inputs[:2])
                    if kind in {"percent","ratio","absolute_percent","free_cash_percent","free_cash_ratio"}:
                        if b<=0:
                            raise ValueError("A positive denominator is required for this interpretation.")
                        base=(a-Decimal(inputs[2]["value"]).copy_abs()) if kind.startswith("free_cash_") else a.copy_abs() if kind=="absolute_percent" else a
                        output=base/b*(100 if kind.endswith("percent") else 1)
                    elif kind=="free_cash":
                        output=a-b.copy_abs()
                    else:
                        output=a-b
                    unit="%" if kind.endswith("percent") else "×" if kind.endswith("ratio") else inputs[0]["unit"]
                    result.update(status="verified",inputs=inputs,raw_output=str(output),
                                  display_output=str(output.quantize(Decimal("0.01"),rounding=ROUND_HALF_EVEN)),unit=unit)
                except ValueError as exc:
                    result["reason"]=str(exc)
                metric["periods"][str(period)]=result
            metrics.append(metric)
    return {"fiscal_year":year,"document_id":report["document_id"],"metrics":metrics,"reconciliations":checks,
            "rounding":"Decimal precision 40; results rounded to two decimal places using ROUND_HALF_EVEN only after calculation",
            "free_cash_flow_definition":"Operating cash flow less absolute reported property/equipment purchases (including intangibles where the filing combines them); a calculated measure, not necessarily the company's definition."}


RATIO_TABS = {
    "Profitability": (("gross_margin", "Gross margin"), ("operating_margin", "Operating margin"),
                      ("net_margin", "Net margin"), ("fcf_margin", "Free cash flow margin")),
    "Financial position": (("equity_assets", "Equity / assets"), ("liabilities_equity", "Liabilities / equity"),
                           ("cash_assets", "Cash / assets"), ("cash_liabilities", "Cash / liabilities")),
    "Cash quality": (("cash_conversion", "Cash conversion"), ("fcf_earnings", "Free cash flow / earnings"),
                     ("capital_intensity", "Capital intensity"), ("operating_cash_margin", "Operating cash margin")),
}


def render_financial_analysis(analysis):
    """Context: show audited ratios in three tabs, with clickable formulas and cited inputs."""
    from html import escape
    import streamlit as st
    year=analysis["fiscal_year"]
    periods=analysis.get("comparison_years",[year,year-1])
    metrics={metric["key"]:metric for metric in analysis["metrics"]}
    checks=analysis["reconciliations"]
    with st.container(border=True,key="ratio-analysis"):
        st.markdown("**Read between the line items.**")
        st.caption("Ratio analysis · "+" vs. ".join(f"FY {period}" for period in periods))
        failed=[check for check in checks if check["status"]=="mismatch"]
        if failed:
            st.warning("Some statement totals do not reconcile. Affected ratios are unavailable; inspect the calculation audit.")
        tabs=st.tabs(list(RATIO_TABS))
        for tab, definitions in zip(tabs,RATIO_TABS.values()):
            with tab:
                columns=st.columns(4)
                for column,(key,label) in zip(columns,definitions):
                    metric=metrics.get(key)
                    with column:
                        with st.popover(label+" ↗",use_container_width=True):
                            st.markdown("**"+label+"**")
                            if metric:
                                st.write(metric["formula"])
                                for period in periods:
                                    result=metric["periods"].get(str(period),{"status":"unavailable"})
                                    st.markdown(f"**FY {period}**")
                                    if result["status"]=="verified":
                                        st.write("Result: "+financial_value(result["display_output"],result["unit"],include_unit=True))
                                        st.table([{"Input":item["key"].replace("_"," "),
                                                   "Value":financial_value(item["value"],item["unit"],include_unit=True)}
                                                  for item in result["inputs"]])
                                        for item in result["inputs"]:
                                            st.caption(item["citation"])
                                            st.text(item["quote"])
                                    else:
                                        st.caption(result.get("reason","Required filing inputs are unavailable."))
                                if key in {"fcf_margin","fcf_earnings","capital_intensity"}:
                                    st.caption(analysis["free_cash_flow_definition"])
                                if key=="cash_conversion":
                                    st.caption("Cash conversion here means operating cash flow / net income; it is not the cash conversion cycle.")
                                if key=="liabilities_equity":
                                    st.caption("Includes all reported liabilities; this is not a debt-only ratio.")
                            else:
                                st.caption("Required filing inputs are unavailable.")
                        result=metric["periods"].get(str(year),{}) if metric else {}
                        display=financial_value(result["display_output"],result["unit"]) if result.get("status")=="verified" else "Unavailable"
                        comparisons=[]
                        for period in periods:
                            if period==year:
                                continue
                            prior=metric["periods"].get(str(period),{}) if metric else {}
                            value=financial_value(prior["display_output"],prior["unit"]) if prior.get("status")=="verified" else "Unavailable"
                            comparisons.append("vs. "+escape(value)+f" in FY {period}")
                        st.markdown('<div class="ratio-card"><div class="ratio-value">'+escape(display)
                                    +'</div><div class="ratio-comparison">'+'<br>'.join(comparisons)+'</div></div>',unsafe_allow_html=True)
        st.markdown('<div class="ratio-footer">Calculated from statement figures. Select any ratio to inspect its formula and source inputs.</div>',unsafe_allow_html=True)
        with st.expander("Statement reconciliation audit"):
            st.caption(f"{sum(check['status']=='passed' for check in checks)} passed; {len(failed)} mismatched; "
                       f"{sum(check['status']=='unavailable' for check in checks)} unavailable. Checks cover recognized rows only.")
            st.json(checks)

"""Context: combine selected filings into year columns without pooling financial amounts."""

from decimal import Decimal
from .calculator import calculate


def year_growth(record,period,year):
    """Context: growth uses the exact displayed year cells and their source quotations."""
    if record["unit"]=="%":
        return None
    current=period_value(record,period,year)
    prior=period_value(record,period-1,year)
    if current is None or prior is None or Decimal(prior)<=0:
        return None
    inputs=[]
    evidence={}
    for fiscal_year,value in ((period,current),(period-1,prior)):
        source=record.get("period_sources",{}).get(fiscal_year,record)
        if source.get("unit")!=record["unit"]:
            return None
        identifier=source["evidence_id"]
        quote=source["quote"]
        # A repeated source ID can have a longer verified quote; retain the full span.
        if len(quote)>len(evidence.get(identifier,"")):
            evidence[identifier]=quote
        inputs.append({"value":value,"currency":record["currency"],
            "scale":"per share" if record["unit"].endswith("per share") else record["unit"].split()[-1],
            "basis":"as reported","scope":record["scope"],"period":f"FY{fiscal_year}",
            "period_type":record.get("period_type","annual"),"metric":record["label"],
            "evidence_id":identifier,"quote":quote})
    try:
        return calculate("growth",inputs,evidence)
    except ValueError:
        return None


def attach_growth(report):
    """Context: export every displayed YoY formula, inputs, and output alongside its row."""
    periods=report["comparison_years"]
    year=max(report["selected_years"])
    for group in ("metrics","balance_sheet","cash_flow","products"):
        report[group]={key:{**record,"growth_by_year":{
            period:year_growth(record,period,year) for period in periods if period-1 in periods}}
            for key,record in report.get(group,{}).items()}
    return report


def combine_reports(reports,selected_years):
    """Context: each cell keeps its source; overlapping periods prefer the newest selected filing."""
    selected=sorted(set(selected_years),reverse=True)
    if not selected:
        raise ValueError("Select at least one fiscal year.")
    latest=selected[0]
    periods=sorted({period for year in selected for period in (year,year-1)},reverse=True)
    result={**reports[latest],"comparison_years":periods,"selected_years":selected}
    if len(selected)==1:
        return attach_growth(result)
    conflicts=[]
    for group in ("metrics","balance_sheet","cash_flow","products"):
        merged={}
        keys=dict.fromkeys(key for year in selected for key in reports[year].get(group,{}))
        for key in keys:
            candidates=[(year,reports[year][group][key]) for year in selected if key in reports[year].get(group,{})]
            base=candidates[0][1]
            cells={}
            sources={}
            for period in periods:
                for filing_year,record in candidates:
                    field="value" if period==filing_year else "prior_value" if period==filing_year-1 else None
                    value=record.get(field) if field else None
                    if value is None or record["unit"]!=base["unit"] or record.get("currency")!=base.get("currency"):
                        continue
                    if period not in cells:
                        cells[period]=value
                        sources[period]={"document_id":record["document_id"],"evidence_id":record["evidence_id"],
                                         "quote":record["quote"],"citation":record["citation"],"unit":record["unit"],
                                         "filing_year":filing_year}
                    elif Decimal(cells[period])!=Decimal(value):
                        conflicts.append({"group":group,"metric":key,"period":period,"chosen_value":cells[period],
                                          "other_value":value,"chosen_document":sources[period]["document_id"],
                                          "other_document":record["document_id"]})
            merged[key]={**base,"year":latest,"value":cells.get(latest),"prior_value":cells.get(latest-1),
                         "period_values":cells,"period_sources":sources}
            # Context: never show an older filing's growth rate as the current year's change.
            if base["year"]!=latest:
                merged[key].pop("change",None)
        result[group]=merged
    analysis={**reports[latest]["analysis"],"comparison_years":periods,"metrics":[]}
    for metric in reports[latest]["analysis"]["metrics"]:
        values={}
        for year in selected:
            source=next(item for item in reports[year]["analysis"]["metrics"] if item["key"]==metric["key"])
            for period,value in source["periods"].items():
                # Keep complete period-specific computations; do not splice ratio operands across filings.
                values.setdefault(period,value)
        analysis["metrics"].append({**metric,"periods":values})
    checks={}
    for year in selected:
        for check in reports[year]["analysis"]["reconciliations"]:
            checks.setdefault((check["name"],check["period"]),check)
    analysis["reconciliations"]=list(checks.values())
    result.update(analysis=analysis,comparison_conflicts=conflicts,
                  comparison_policy="Newest selected filing supplies overlapping periods; incompatible units are unavailable.",
                  source_reports={year:reports[year] for year in selected})
    return attach_growth(result)


def period_value(record,period,year):
    """Context: one lookup supports both single-filing and combined-year tables and charts."""
    if "period_values" in record:
        return record["period_values"].get(period)
    return record.get("value") if period==year else record.get("prior_value") if period==year-1 else None

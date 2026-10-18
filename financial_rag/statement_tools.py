"""Context: supply source-parsed statement observations to the LLM and calculator.

Only retained retrieval evidence is parsed. No reference answers, external sources
or company-specific amounts enter this stage. Unsupported layouts fail closed.
"""

import re

from .calculator import calculate
from .core import Chunk
from .overview import read_row
from .retrieval import primary_balance_statement


def prepare_statement_tools(question,evidence,sources):
    """Context: resolve exact units/year columns and component metadata before generation."""
    if not re.search(r"\bbalance\s+sheets?\b",question,re.I):
        return {"observations":[],"calculator_requests":[],"calculator_results":[]}
    definitions=[]
    if re.search(r"\bcommercial paper\b",question,re.I):
        definitions.append(("Commercial paper",None))
    if re.search(r"\binventor(?:y|ies)\b",question,re.I):
        definitions.append(("Inventories",None))
    if re.search(r"\baccounts receivable\b",question,re.I):
        definitions.append(("Accounts receivable, net",None))
    if re.search(r"\bterm debt\b",question,re.I):
        definitions.extend(("Term debt",component) for component in ("Current liabilities","Non-current liabilities"))
    years=sorted({int(y) for y in re.findall(r"(?<!\d)(?:19|20)\d{2}\b",question)})
    observations=[];seen=set()
    for key,text in evidence.items():
        source=sources[key]
        if not primary_balance_statement(text):
            continue
        currency_codes=set(re.findall(r"\b(?:USD|EUR|GBP|CAD|JPY|CNY|AUD|CHF|INR)\b",text))
        currency=next(iter(currency_codes)) if len(currency_codes)==1 else "USD" if source.get("ticker")=="AAPL" else None
        if currency is None:
            continue
        chunk=Chunk(key,key,source["document_id"],text,source["ticker"],source["fiscal_year"],
                    "FY",source["doc_type"],source["section_path"],source["location"],source["source"])
        for label,component in definitions:
            for year in years:
                # Context: requests to cite both filings use each filing's OWN
                # year, rather than sourcing both operands from one comparative table.
                if re.search(r"cite\s+both\s+filings",question,re.I) and chunk.fiscal_year!=year:
                    continue
                record=read_row(chunk,label,year,"USD millions",currency,component=component)
                identity=(label,component,year)
                if not record or identity in seen:
                    continue
                scale=record["unit"].split(" ",1)[1]
                if re.search(r"\bUSD millions\b",question,re.I) and (currency!="USD" or scale!="millions"):
                    continue
                observation={"value":record["value"],"currency":currency,"scale":scale,
                    "basis":"reported","scope":record["scope"],"period":f"FY{year}","period_type":"year_end",
                    "metric":label,"evidence_id":key,
                    "quote":record["component_quote"] if component else record["row_quote"],
                    "period_quote":record["period_quote"]}
                if component:
                    observation["component"]=component
                observations.append(observation);seen.add(identity)
    requests=[]
    if len(years)==2 and re.search(r"\b(?:percentage|percent)\s+(?:change|increase|growth)\b",question,re.I):
        for label,component in definitions:
            items=[o for o in observations if o["metric"]==label and not component]
            prior=next((o for o in items if o["period"]==f"FY{years[0]}"),None)
            current=next((o for o in items if o["period"]==f"FY{years[1]}"),None)
            if prior and current:
                requests.extend({"operation":op,"inputs":[current,prior]} for op in ("difference","growth"))
    if re.search(r"\bterm debt\b",question,re.I) and re.search(r"\b(?:adding|sum|total)\b",question,re.I):
        for year in years:
            parts=[o for o in observations if o["metric"]=="Term debt" and o["period"]==f"FY{year}"]
            if len(parts)==2:
                requests.append({"operation":"sum","inputs":parts})
    results=[{key:value for key,value in calculate(r["operation"],r["inputs"],evidence).items()
              if key!="inputs"} for r in requests]
    return {"observations":observations,"calculator_requests":requests,"calculator_results":results}

"""Context: Decimal arithmetic operates on cited, explicitly comparable observations.

No eval(), arbitrary expressions, or executable model-generated code is allowed.
"""

from decimal import Decimal, InvalidOperation, localcontext
import re


def canonical_quote(quote, passage):
    """Context: accept whitespace-only PDF differences and return the ORIGINAL span.

    Punctuation, digits, word order, and ellipses remain literal. Never use fuzzy
    matching or manufacture a quote by joining noncontiguous source fragments.
    """
    if not isinstance(quote, str) or not quote.strip() or not isinstance(passage, str):
        raise ValueError("Quote is empty or invalid.")
    parts = re.findall(r"\S+", quote)
    match = re.search(r"\s+".join(re.escape(part) for part in parts), passage)
    if match is None:
        raise ValueError("Quote is not present in retrieved evidence.")
    return match.group()


def table_quote_with_header(quote, passage, years):
    """Context: expand a verified row quote to its nearest original table header.

    Only missing year numerals are eligible. Never borrow a filing year, a later
    header, or a year from an older table when the nearest header disagrees.
    Returning the original contiguous source span makes this expansion auditable.
    """
    start = passage.find(quote)
    if start < 0:
        return quote
    offset, headers = 0, []
    for line in passage[:start].splitlines(keepends=True):
        header_years = {int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b",line)}
        remainder = re.sub(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december|fiscal|years?|ended|ending|change)\b", "",line,flags=re.I)
        if len(header_years) >= 2 and re.fullmatch(r"[\d\s,|%.$()/\u2014\u2013:-]*",remainder):
            headers.append((offset,header_years))
        offset += len(line)
    if headers and {int(year) for year in years} <= headers[-1][1]:
        return passage[headers[-1][0]:start+len(quote)]
    return quote


def numeric_literals(text):
    """Context: numerical validation ignores thousands separators, not signs."""
    # Fiscal-year abbreviations must not hide a year from numeric grounding checks.
    text = re.sub(r"\bFY\s*(?=\d{4}\b)", "", text, flags=re.I)
    result = set()
    for match in re.finditer(r"(?<![\w.])\(?[-+]?\d[\d,]*(?:\.\d+)?\)?", text):
        token = match.group().replace(",", "")
        if token.startswith("(") and token.endswith(")"):
            token = "-" + token[1:-1]
        try:
            result.add(Decimal(token))
        except InvalidOperation:
            continue
    return result


def date_literals(text):
    """Context: identify only explicitly written dates and fiscal years in a claim."""
    matches=re.findall(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",text,re.I)
    matches+=re.findall(r"\b(?:FY\s*|fiscal(?:\s+year)?\s+|in\s+|for\s+)((?:19|20)\d{2})\b",text,re.I)
    return numeric_literals(" ".join(matches))


def sentence_quote_with_dates(quote, passage, missing):
    """Context: restore a literal sentence's date, never a number from another sentence.

    Expansion is allowed only for already verified quotations and one unambiguous
    original sentence containing every missing date token. No fuzzy matching,
    synthesized wording, document-label inference or cross-sentence joins occur.
    """
    canonical_quote(quote,passage)
    candidates=[]
    for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])|\n\s*\n",passage):
        # Context: a block of table rows is not a narrative sentence. In
        # particular, never use this path to bypass the nearest-header rule.
        if (len(sentence)>1600 or not sentence.rstrip().endswith((".","!","?")) or
            re.search(r"<table>|\|",sentence,re.I) or
            not re.search(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",sentence,re.I)):
            continue
        try:
            canonical_quote(quote,sentence)
        except ValueError:
            continue
        if missing<=numeric_literals(sentence):
            candidates.append(sentence)
    return candidates[0] if len(candidates)==1 else quote


def calculate(operation, inputs, evidence):
    """Context: values must occur in quoted evidence; metadata must be compatible.

    Metadata supplied by an extractor still needs domain review. Presence of a
    number in a quote does not prove that it is the correct table cell.
    """
    if operation not in {"growth", "margin", "difference", "sum"} or len(inputs) != 2:
        raise ValueError("Supported operations: growth, margin, difference, sum; exactly two inputs.")
    required = {"value", "currency", "scale", "basis", "scope", "period", "period_type",
                "metric", "evidence_id", "quote"}
    for item in inputs:
        if not required <= item.keys():
            raise ValueError("Calculation operand lacks required provenance/comparability metadata.")
        if any(not str(item[key]).strip() or str(item[key]).lower() in {"unknown", "none", "n/a"}
               for key in required):
            raise ValueError("Calculation metadata must be explicit, not unknown.")
        if not Decimal(item["value"]).is_finite():
            raise ValueError("Calculation inputs must be finite numbers.")
        if item["evidence_id"] not in evidence:
            raise ValueError("Calculation quote is not present in retrieved evidence.")
        try:
            item["quote"] = canonical_quote(item["quote"], evidence[item["evidence_id"]])
        except ValueError as exc:
            raise ValueError("Calculation quote is not present in retrieved evidence.") from exc
        if Decimal(item["value"]) not in numeric_literals(item["quote"]):
            raise ValueError("Calculation input value does not appear in its quote.")
        if item.get("period_quote"):
            item["period_quote"]=canonical_quote(item["period_quote"],evidence[item["evidence_id"]])
            years=re.findall(r"(?<!\d)(?:19|20)\d{2}\b",item["period"])
            if not years or not all(Decimal(y) in numeric_literals(item["period_quote"]) for y in years):
                raise ValueError("Calculation period must occur in its cited year header.")
    a, b = inputs
    for key in ("currency", "scale", "basis", "scope", "period_type"):
        if a[key] != b[key]:
            raise ValueError(f"Incompatible calculation inputs: {key}.")
    if operation in {"growth", "difference"} and a["metric"] != b["metric"]:
        raise ValueError("Comparison requires the same metric.")
    if operation in {"growth", "difference"} and a["period"] == b["period"]:
        raise ValueError("Period comparison requires different periods.")
    if operation == "margin" and a["period"] != b["period"]:
        raise ValueError("Margin numerator and denominator require the same period.")
    if operation == "sum":
        # Context: two parts of the same measure may be added only for the same
        # period, with distinct source-backed component labels (not a total plus a part).
        if a["period"]!=b["period"] or a["metric"]!=b["metric"]:
            raise ValueError("Sum requires the same period and metric.")
        labels=[]
        for item in inputs:
            component=item.get("component")
            if not isinstance(component,str) or not component.strip():
                raise ValueError("Sum requires explicit distinct component labels.")
            label=re.sub(r"[^a-z0-9]+"," ",component.casefold()).strip()
            quoted=re.sub(r"[^a-z0-9]+"," ",item["quote"].casefold())
            if not label or not re.search(r"\b"+re.escape(label)+r"\b",quoted) or re.search(r"\btotal\b",label):
                raise ValueError("Sum component label must occur in its quote and identify a part, not a total.")
            labels.append(label)
        if labels[0]==labels[1]:
            raise ValueError("Sum components must be distinct; do not count a part twice.")
        for index,item in enumerate(inputs):
            # Context: the amount must be on this metric's row UNDER its own
            # component heading, rather than borrowed from the other component.
            heading=r"^\s*"+r"[\W_]+".join(map(re.escape,labels[index].split()))+r"\b[^\n]*\n"
            match=re.search(heading,item["quote"],re.I|re.M)
            if match is None:
                raise ValueError("Sum quote must include the original component heading and metric row.")
            section=item["quote"][match.end():]
            other=r"^\s*"+r"[\W_]+".join(map(re.escape,labels[1-index].split()))+r"\b"
            boundary=re.search(other,section,re.I|re.M)
            if boundary:
                section=section[:boundary.start()]
            metric=r"^\s*"+r"[\W_]+".join(map(re.escape,re.findall(r"\w+",item["metric"])))+r"\b"
            if not any(re.search(metric,line,re.I) and Decimal(item["value"]) in numeric_literals(line)
                       for line in section.splitlines()):
                raise ValueError("Sum amount must occur on the metric row within its cited component section.")
    with localcontext() as context:
        context.prec = 40
        first, second = Decimal(a["value"]), Decimal(b["value"])
        if operation in {"growth","margin"} and second == 0:
            raise ValueError("Cannot divide by zero.")
        if operation == "growth" and second < 0:
            raise ValueError("Negative growth denominator requires a separately defined interpretation.")
        if operation == "growth":
            result, formula = (first - second) / second * 100, "(current - prior) / prior * 100"
        elif operation == "margin":
            result, formula = first / second * 100, "numerator / denominator * 100"
        elif operation == "sum":
            result, formula = first + second, "first component + second component"
        else:
            result, formula = first - second, "current - prior"
        return {"operation": operation, "formula": formula, "inputs": inputs,
                "raw_output": str(result), "display_output": str(result.quantize(Decimal("0.01"))),
                "unit": "%" if operation in {"growth", "margin"} else a["scale"]}

"""Context: curated retrieval vocabulary, not an accounting equivalence engine.

Aliases find candidate disclosures. They never authorize substituting measures,
converting units, calculating values, or changing literal source quotations.
"""

from dataclasses import dataclass
import re


TAXONOMY_VERSION = "financial-terms-v2"


@dataclass(frozen=True)
class Concept:
    """Context: separate question vocabulary from labels eligible for row ranking."""
    key: str
    category: str
    aliases: tuple[str, ...]
    rows: tuple[str, ...]
    kind: str = "amount"
    caution: str = "Verify issuer definition, scope, period, currency and units."


def concept(key, category, aliases, *, rows=None, kind="amount", caution=None):
    """Context: an empty row list explicitly disables numeric row boosting."""
    labels = tuple(part.strip() for part in aliases.split("|"))
    return Concept(key, category, labels, labels if rows is None else tuple(rows), kind,
                   caution or Concept.__dataclass_fields__["caution"].default)


CONCEPTS = (
    concept("revenue", "Income statement", "total net sales|total revenues|total revenue|net revenues|net revenue|net sales|revenues|revenue|sales|turnover",
            caution="Gross/net presentation, product/segment scope and issuer meaning of turnover must be verified."),
    concept("cost_sales", "Income statement", "total cost of sales|cost of sales|cost of revenues|cost of revenue|total cost of revenue|cost of goods sold|COGS"),
    concept("gross_profit", "Income statement", "gross profit|gross profit amount", rows=("gross profit", "gross margin"),
            caution="Gross margin can be an amount or a percentage. Check the row units."),
    concept("gross_margin", "Ratios", "gross margin percentage|gross profit margin|gross margin percent|gross margin %", rows=("gross margin percentage", "gross margin percent", "gross margin", "total gross margin"), kind="percentage"),
    concept("research_development", "Income statement", "research and development|R&D|research and development expenses"),
    concept("selling_general_admin", "Income statement", "selling general and administrative|selling general and administrative expenses|SG&A|sales general and administrative"),
    concept("operating_expenses", "Income statement", "total operating expenses|operating expenses|OPEX"),
    concept("operating_income", "Income statement", "operating income|income from operations|operating profit|operating income (loss)",
            caution="EBIT and EBITDA are separate concepts; do not substitute adjusted operating profit."),
    concept("pretax_income", "Income statement", "income before income taxes|income before provision for income taxes|earnings before income taxes|profit before tax|pretax income|pre-tax income"),
    concept("income_tax", "Income statement", "provision for income taxes|income tax expense|income taxes|tax expense"),
    concept("net_income", "Income statement", "net income|net earnings|net profit|net income (loss)|profit for the year",
            caution="Consolidated net income differs from income attributable to owners or common shareholders."),
    concept("basic_eps", "Per-share", "basic earnings per share|basic EPS|basic net income per share", rows=("basic earnings per share", "basic net income per share"), kind="per_share"),
    concept("diluted_eps", "Per-share", "diluted earnings per share|diluted EPS|diluted net income per share", rows=("diluted earnings per share", "diluted net income per share"), kind="per_share",
            caution="Do not treat a bare Diluted row as EPS: it may be a weighted-average share count."),
    concept("cash_equivalents", "Balance sheet", "cash and cash equivalents|cash equivalents|cash & cash equivalents", rows=("cash and cash equivalents",), caution="Restricted cash and short-term investments are separate balances."),
    concept("receivables", "Balance sheet", "accounts receivable net|accounts receivable|trade receivables|trade debtors"),
    concept("inventory", "Balance sheet", "inventories|inventory", caution="Inventory turnover is a ratio, not the inventory balance."),
    concept("current_assets", "Balance sheet", "total current assets|current assets"),
    concept("ppe", "Balance sheet", "property plant and equipment net|property and equipment net|PP&E|PPE", rows=("property plant and equipment net", "property and equipment net"), caution="Net carrying value is not gross asset cost or capital expenditure."),
    concept("goodwill", "Balance sheet", "goodwill"),
    concept("total_assets", "Balance sheet", "total assets"),
    concept("payables", "Balance sheet", "accounts payable|trade payables|trade creditors"),
    concept("current_liabilities", "Balance sheet", "total current liabilities|current liabilities"),
    concept("commercial_paper", "Balance sheet", "commercial paper", caution="Use the balance-sheet carrying amount when exact statement units are requested; commentary may round to billions."),
    concept("term_debt", "Balance sheet", "term debt", caution="Current and non-current portions are distinct components; exclude commercial paper from their sum."),
    concept("long_term_debt", "Balance sheet", "long-term debt|long term borrowings|non-current borrowings", caution="Check current maturities, lease liabilities and net/gross carrying amount."),
    concept("total_liabilities", "Balance sheet", "total liabilities"),
    concept("equity", "Balance sheet", "total shareholders equity|total stockholders equity|shareholders equity|stockholders equity|total equity", caution="Parent equity and total equity including noncontrolling interests differ."),
    concept("retained_earnings", "Balance sheet", "retained earnings|accumulated deficit|retained earnings (accumulated deficit)", caution="An accumulated deficit is a negative retained balance, not current-year loss."),
    concept("operating_cash", "Cash flow", "operating cash flow|cash flow from operations|cash from operations|net cash provided by operating activities|net cash used in operating activities|net cash provided by (used in) operating activities|cash generated by operating activities|CFO|OCF",
            caution="Preserve provided/used signs; CFO may mean chief financial officer in narrative text."),
    concept("investing_cash", "Cash flow", "investing cash flow|net cash used in investing activities|net cash provided by investing activities|net cash provided by (used in) investing activities|cash used in investing activities|cash generated by investing activities"),
    concept("financing_cash", "Cash flow", "financing cash flow|net cash used in financing activities|net cash provided by financing activities|net cash provided by (used in) financing activities|cash used in financing activities|cash generated by financing activities"),
    concept("capex", "Cash flow", "capital expenditures|capital expenditure|CAPEX|purchases of property and equipment|payments for acquisition of property plant and equipment|additions to property and equipment",
            caution="Cash purchases, accrued additions and purchases including intangibles can have different scope."),
    concept("depreciation_amortization", "Cash flow", "depreciation and amortization|D&A"),
    concept("share_compensation", "Cash flow", "share-based compensation|stock-based compensation|share-based compensation expense|stock-based compensation expense|SBC"),
    concept("dividends_paid", "Cash flow", "dividends paid|payments for dividends and dividend equivalents|payments of dividends"),
    concept("free_cash_flow", "Derived / issuer-defined", "free cash flow|FCF", rows=("free cash flow",), caution="Issuer definitions vary. Use a disclosed reconciliation or an explicitly specified audited formula."),
    concept("adjusted_ebitda", "Adjusted / non-GAAP", "adjusted EBITDA|normalized EBITDA|underlying EBITDA", caution="Not interchangeable with EBITDA. Match the issuer's exact reconciliation and definition."),
    concept("adjusted_net_income", "Adjusted / non-GAAP", "adjusted net income|adjusted net earnings|non-GAAP net income", caution="Not interchangeable with reported net income; verify excluded items."),
    concept("adjusted_eps", "Adjusted / non-GAAP", "adjusted diluted EPS|adjusted EPS|non-GAAP EPS|adjusted earnings per share", kind="per_share", caution="Confirm basic versus diluted and the issuer's adjustment reconciliation."),
    concept("ebit", "Derived / issuer-defined", "EBIT|earnings before interest and taxes", caution="Do not assume EBIT equals operating income; verify the reconciliation."),
    concept("ebitda", "Derived / issuer-defined", "EBITDA|earnings before interest taxes depreciation and amortization", caution="Do not equate EBITDA with adjusted EBITDA or cash from operations."),
    concept("operating_margin", "Ratios", "operating margin|operating profit margin", kind="percentage"),
    concept("net_margin", "Ratios", "net profit margin|net income margin|net margin", kind="percentage"),
    concept("gross_margin_ambiguous", "Ambiguous", "gross margin", rows=(), kind="ambiguous", caution="Could mean gross profit amount or gross margin percentage. Resolve using the question and disclosed units."),
    concept("current_ratio", "Ratios", "current ratio", rows=(), kind="ratio"),
    concept("quick_ratio", "Ratios", "quick ratio|acid-test ratio", rows=(), kind="ratio", caution="Quick-asset definitions vary; do not substitute the current ratio."),
    concept("working_capital", "Derived / issuer-defined", "working capital|net working capital", rows=("working capital", "net working capital"), caution="Operating working capital may exclude cash and debt; confirm scope."),
    concept("inventory_turnover", "Ratios", "inventory turnover|inventory turns", rows=(), kind="ratio"),
    concept("debt_equity", "Ratios", "debt to equity|debt-to-equity ratio", rows=(), kind="ratio", caution="Debt differs from total liabilities; confirm which debt and equity balances are used."),
    concept("return_equity", "Ratios", "return on equity|ROE", rows=(), kind="percentage", caution="Check average versus closing equity and attributable income."),
    concept("return_assets", "Ratios", "return on assets|ROA", rows=(), kind="percentage"),
    concept("income_statement", "Statement titles", "income statement|statement of operations|statements of operations|statement of earnings|statement of profit or loss|profit and loss statement|P&L", rows=(), kind="document"),
    concept("balance_sheet", "Statement titles", "balance sheet|balance sheets|statement of financial position", rows=(), kind="document"),
    concept("cash_flow_statement", "Statement titles", "cash flow statement|statement of cash flows|statements of cash flows", rows=(), kind="document"),
    concept("yoy", "Period / comparison", "year over year|year-on-year|YoY|year-over-year", rows=(), kind="comparison"),
    concept("ytd", "Period / comparison", "year to date|YTD|year-to-date", rows=(), kind="period"),
    concept("ttm", "Period / comparison", "trailing twelve months|TTM|last twelve months|LTM", rows=(), kind="period", caution="A rolling twelve-month period need not equal a fiscal year."),
    concept("percentage_points", "Period / comparison", "percentage points|percentage point|basis points|basis point|bps", rows=(), kind="comparison", caution="Percentage-point changes and relative percentage growth are different calculations; 100 basis points equals one percentage point."),
)
BY_KEY = {item.key: item for item in CONCEPTS}


def normalize(text):
    """Context: punctuation variants match while original evidence stays untouched."""
    return re.sub(r"[^a-z0-9%]+", " ", text.casefold().replace("&", " and ").replace("’", "'").replace("'", "")).strip()


def match_concepts(query):
    """Context: longer phrases suppress embedded generic terms; retain disjoint requests."""
    text = normalize(query)
    matches = []
    for item in CONCEPTS:
        for alias in item.aliases:
            for match in re.finditer(r"(?<!\w)" + re.escape(normalize(alias)) + r"(?!\w)", text):
                matches.append((match.start(), match.end(), item))
    accepted, occupied = [], []
    for start, end, item in sorted(matches, key=lambda value: (-(value[1]-value[0]), value[0])):
        if any(start < right and end > left for left, right in occupied):
            continue
        occupied.append((start, end))
        if item not in accepted:
            accepted.append(item)
    return accepted


def has_numeric_row(text, item):
    """Context: rank candidate numeric rows without interpreting their financial value."""
    for line in text.splitlines():
        # Context: Gross margin is an ambiguous label; a percentage row must not
        # stand in for gross profit. Other amount rows can include YoY % columns.
        if item.key == "gross_profit" and "%" in line:
            continue
        row = normalize(line)
        if any(re.match(r"^(?:total )?" + re.escape(normalize(label))
                        + r" (?:\d{1,2} )?\d", row) for label in item.rows):
            return True
    return False

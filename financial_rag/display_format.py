"""Context: presentation conventions shared by all financial dashboard panels.

Keep source currency and scale; never convert units or alter calculation inputs.
"""

from decimal import Decimal
from html import escape


def amount(value,places=None):
    """Context: grouping and accounting parentheses affect display only."""
    if value is None:
        return "Unavailable"
    number=Decimal(value)
    magnitude=number.copy_abs()
    text=f"{magnitude:,.{places}f}" if places is not None else f"{magnitude:,f}"
    if places is None and "." in text:
        text=text.rstrip("0").rstrip(".")
    return f"({text})" if number<0 else text


def percentage(value,signed=False):
    """Context: percentages consistently use two decimal places; changes keep their sign."""
    return "Unavailable" if value is None else format(Decimal(value),"+.2f" if signed else ".2f")+"%"


def financial_value(value,unit,include_unit=False):
    """Context: monetary units belong in headings; EPS and ratios keep explicit distinctions."""
    if value is None:
        return "Unavailable"
    if unit=="%":
        return percentage(value)
    if unit=="×":
        return amount(value,2)+"×"
    text=amount(value,2 if unit.endswith("per share") else None)
    return text+" "+unit if include_unit else text


def panel_heading(title,unit):
    """Context: one consistent top-right currency/scale treatment throughout the app."""
    return '<div class="statement-heading"><b>'+escape(title)+'</b><span class="reporting-unit">'+escape(unit)+'</span></div>'

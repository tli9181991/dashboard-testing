"""Fundamental data for one symbol, in a shape the UI and the LLM can both use.

yfinance's ``.info`` is a loose dict whose keys come and go by ticker, listing venue
and time of day. Everything here treats a missing field as missing rather than as
zero, because a blank price-to-book and a price-to-book of 0.0 mean very different
things and quietly turning one into the other is how a screen ends up recommending
a company with no equity.

``FundamentalSnapshot.to_prompt_text`` renders the same numbers as plain text for
the assistant, so the model reasons over what the user is looking at rather than
over whatever it remembers about the company.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

SECTIONS: list[tuple[str, list[tuple[str, str, str]]]] = [
    ("Valuation", [
        ("trailingPE", "Trailing P/E", "ratio"),
        ("forwardPE", "Forward P/E", "ratio"),
        ("pegRatio", "PEG", "ratio"),
        ("priceToBook", "Price / book", "ratio"),
        ("priceToSalesTrailing12Months", "Price / sales", "ratio"),
        ("enterpriseToEbitda", "EV / EBITDA", "ratio"),
    ]),
    ("Profitability", [
        ("grossMargins", "Gross margin", "pct"),
        ("operatingMargins", "Operating margin", "pct"),
        ("profitMargins", "Net margin", "pct"),
        ("returnOnEquity", "Return on equity", "pct"),
        ("returnOnAssets", "Return on assets", "pct"),
    ]),
    ("Growth", [
        ("revenueGrowth", "Revenue growth (yoy)", "pct"),
        ("earningsGrowth", "Earnings growth (yoy)", "pct"),
        ("earningsQuarterlyGrowth", "Earnings growth (qoq)", "pct"),
    ]),
    ("Balance sheet", [
        ("totalCash", "Cash", "money"),
        ("totalDebt", "Debt", "money"),
        ("debtToEquity", "Debt / equity", "ratio"),
        ("currentRatio", "Current ratio", "ratio"),
        ("quickRatio", "Quick ratio", "ratio"),
        ("freeCashflow", "Free cash flow", "money"),
    ]),
    ("Dividend", [
        ("dividendYield", "Dividend yield", "pct"),
        ("payoutRatio", "Payout ratio", "pct"),
    ]),
    ("Market", [
        ("marketCap", "Market cap", "money"),
        ("beta", "Beta", "ratio"),
        ("fiftyTwoWeekHigh", "52-week high", "price"),
        ("fiftyTwoWeekLow", "52-week low", "price"),
        ("averageVolume", "Average volume", "count"),
    ]),
]

#: yfinance reports some ratios as fractions and some as percentages already.
#: debtToEquity comes back as a percentage (e.g. 45.2 meaning 45.2%), which is why
#: it is a ratio here and not a pct.
ANALYST_FIELDS = [
    ("recommendationKey", "Consensus"),
    ("numberOfAnalystOpinions", "Analysts covering"),
    ("targetMeanPrice", "Mean target"),
    ("targetHighPrice", "High target"),
    ("targetLowPrice", "Low target"),
]


def _clean(value: Any) -> Optional[float]:
    """Numeric value, or None. Strings, NaN and infinities all read as missing."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def format_value(value: Optional[float], kind: str) -> str:
    if value is None:
        return "—"
    if kind == "pct":
        return f"{value * 100:,.1f}%"
    if kind == "money":
        for cut, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
            if abs(value) >= cut:
                return f"${value / cut:,.2f}{suffix}"
        return f"${value:,.0f}"
    if kind == "count":
        for cut, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
            if abs(value) >= cut:
                return f"{value / cut:,.1f}{suffix}"
        return f"{value:,.0f}"
    if kind == "price":
        return f"${value:,.2f}"
    return f"{value:,.2f}"


@dataclass
class FundamentalSnapshot:
    symbol: str
    name: str = ""
    sector: str = ""
    industry: str = ""
    currency: str = ""
    price: Optional[float] = None
    sections: dict[str, list[tuple[str, Optional[float], str]]] = field(default_factory=dict)
    analysts: dict[str, Any] = field(default_factory=dict)
    business_summary: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.sections)

    @property
    def coverage(self) -> float:
        """Share of the tracked fields the vendor actually returned."""
        total = sum(len(rows) for rows in self.sections.values())
        if not total:
            return 0.0
        present = sum(1 for rows in self.sections.values() for _, v, _ in rows if v is not None)
        return present / total

    def rows(self, section: str) -> list[tuple[str, str]]:
        return [(label, format_value(value, kind))
                for label, value, kind in self.sections.get(section, [])]

    def get(self, section: str, label: str) -> Optional[float]:
        for row_label, value, _ in self.sections.get(section, []):
            if row_label == label:
                return value
        return None

    def upside(self) -> Optional[float]:
        """Fraction between the last price and the mean analyst target."""
        target = _clean(self.analysts.get("targetMeanPrice"))
        if target is None or not self.price:
            return None
        return target / self.price - 1

    def to_prompt_text(self) -> str:
        """The same numbers as plain text, for the assistant's context."""
        if not self.ok:
            return f"Fundamentals for {self.symbol}: unavailable ({self.error or 'no data'})."

        lines = [f"Fundamentals for {self.symbol} ({self.name or 'unknown name'})"]
        if self.sector or self.industry:
            lines.append(f"Sector: {self.sector or '—'} / {self.industry or '—'}")
        if self.price:
            lines.append(f"Last price: {format_value(self.price, 'price')}")

        for section, _ in SECTIONS:
            pairs = [f"{label} {value}" for label, value in self.rows(section) if value != "—"]
            if pairs:
                lines.append(f"{section}: " + "; ".join(pairs))

        if self.analysts:
            bits = []
            for key, label in ANALYST_FIELDS:
                value = self.analysts.get(key)
                if value in (None, ""):
                    continue
                if key == "recommendationKey":
                    bits.append(f"{label} {str(value).replace('_', ' ')}")
                elif key == "numberOfAnalystOpinions":
                    bits.append(f"{label} {int(value)}")
                else:
                    bits.append(f"{label} {format_value(_clean(value), 'price')}")
            up = self.upside()
            if up is not None:
                bits.append(f"Implied upside to mean target {up * 100:+.1f}%")
            if bits:
                lines.append("Analysts: " + "; ".join(bits))

        lines.append(f"(Vendor supplied {self.coverage:.0%} of the tracked fields.)")
        return "\n".join(lines)


def from_info(symbol: str, info: dict, price: Optional[float] = None) -> FundamentalSnapshot:
    """Build a snapshot from an already-fetched yfinance info dict."""
    if not info:
        return FundamentalSnapshot(symbol=symbol, error="no data returned")

    sections: dict[str, list[tuple[str, Optional[float], str]]] = {}
    for section, fields in SECTIONS:
        sections[section] = [(label, _clean(info.get(key)), kind) for key, label, kind in fields]

    last = price if price is not None else _clean(
        info.get("currentPrice") or info.get("regularMarketPrice")
    )

    return FundamentalSnapshot(
        symbol=symbol,
        name=str(info.get("longName") or info.get("shortName") or ""),
        sector=str(info.get("sector") or ""),
        industry=str(info.get("industry") or ""),
        currency=str(info.get("currency") or ""),
        price=last,
        sections=sections,
        analysts={key: info.get(key) for key, _ in ANALYST_FIELDS if info.get(key) not in (None, "")},
        business_summary=str(info.get("longBusinessSummary") or ""),
    )


def fetch(symbol: str) -> FundamentalSnapshot:
    """Fetch fundamentals for one symbol. Never raises."""
    try:
        import yfinance as yf
    except ImportError:
        return FundamentalSnapshot(symbol=symbol, error="yfinance is not installed")

    try:
        info = yf.Ticker(symbol).info or {}
    except Exception as exc:
        return FundamentalSnapshot(symbol=symbol, error=f"lookup failed: {exc}")

    snapshot = from_info(symbol, info)
    if not snapshot.sections:
        snapshot.error = "no fundamental fields returned"
    return snapshot

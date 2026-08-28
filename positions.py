"""Reading portfolio state off disk.

``portfolio.csv`` overloads ``total_amount``: for crypto it held a dollar amount,
for equities a share count. That ambiguity is a live-money hazard — it makes
position caps and PnL asset-class dependent — so everything downstream now works
in explicit unit counts.

Existing files keep working. Add a ``quantity`` column and it is used directly;
otherwise the legacy column is converted under the old convention and the
conversion is reported, so nothing is reinterpreted silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from strategy import AssetClass, Position

PORTFOLIO_FILE = Path("portfolio.csv")
TRUTHY = {"true", "1", "yes", "y", "t"}


@dataclass(frozen=True)
class Holding:
    symbol: str
    position: Position
    asset_class: AssetClass
    #: Set when quantity was derived from the legacy dollar-denominated column.
    converted_from_notional: bool = False

    @property
    def quantity(self) -> float:
        return self.position.quantity


def _as_bool(value) -> bool:
    return str(value).strip().lower() in TRUTHY


def load_portfolio(path: Optional[Path] = None) -> dict[str, Holding]:
    """Load holdings keyed by symbol. Returns {} when the file is absent or empty."""
    path = Path(path) if path is not None else PORTFOLIO_FILE
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    if df.empty:
        return {}

    # Tolerate the historical misspelling.
    if "total_ammount" in df.columns and "total_amount" not in df.columns:
        df = df.rename(columns={"total_ammount": "total_amount"})

    holdings: dict[str, Holding] = {}
    for _, row in df.iterrows():
        symbol = str(row["ticker"]).strip().upper()
        if not symbol:
            continue

        asset_class = AssetClass.infer(symbol)
        avg_price = float(row.get("averaged_price", 0.0) or 0.0)
        converted = False

        if "quantity" in df.columns and pd.notna(row.get("quantity")):
            quantity = float(row["quantity"])
        else:
            legacy = float(row.get("total_amount", 0.0) or 0.0)
            if asset_class is AssetClass.CRYPTO and avg_price > 0:
                # Legacy convention: crypto rows stored dollars, not coins.
                quantity = legacy / avg_price
                converted = True
            else:
                quantity = legacy

        holdings[symbol] = Holding(
            symbol=symbol,
            position=Position(
                quantity=quantity,
                avg_price=avg_price,
                long_term=_as_bool(row.get("long_term", False)),
            ),
            asset_class=asset_class,
            converted_from_notional=converted,
        )
    return holdings


def conversion_notes(holdings: dict[str, Holding]) -> list[str]:
    """Human-readable notes for any row whose quantity had to be inferred."""
    notes = []
    for h in holdings.values():
        if h.converted_from_notional:
            notes.append(
                f"{h.symbol}: read total_amount as ${h.position.avg_price * h.quantity:,.2f} "
                f"notional -> {h.quantity:.8f} units. Add a 'quantity' column to make this explicit."
            )
    return notes

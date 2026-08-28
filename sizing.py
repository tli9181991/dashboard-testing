"""Volatility-targeted position sizing.

Replaces the fixed ``stock_share_size = 6`` with a size that falls as a name gets
noisier, so every open position contributes a comparable amount of risk instead
of a comparable number of shares. A $30 utility and a $300 semiconductor sized at
six shares each are not remotely the same bet.

    target_notional = equity * target_vol / annualised_vol(symbol)

capped by ``max_position_pct`` so a very quiet name cannot swallow the book.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from strategy import AssetClass

TRADING_DAYS = 252


@dataclass(frozen=True)
class SizingParams:
    #: Annualised volatility budget per position, e.g. 0.15 = 15%.
    target_vol: float = 0.15
    #: Hard cap on one position as a fraction of equity.
    max_position_pct: float = 0.25
    #: Floor on the vol estimate, so a near-zero reading cannot imply huge size.
    min_vol: float = 0.05
    #: Ceiling on the vol estimate, keeping very noisy names to a small stub.
    max_vol: float = 2.00
    #: Smallest crypto increment worth transacting.
    crypto_precision: int = 8


def annualized_vol_from_atr(atr: float, price: float, periods: int = TRADING_DAYS) -> float:
    """ATR as a fraction of price, annualised.

    ATR is used rather than close-to-close standard deviation because it accounts
    for gaps, which is what actually hurts a stop-based strategy.
    """
    if price <= 0 or atr <= 0 or not math.isfinite(atr) or not math.isfinite(price):
        return 0.0
    return (atr / price) * math.sqrt(periods)


def target_quantity(
    equity: float,
    price: float,
    ann_vol: float,
    asset_class: AssetClass = AssetClass.EQUITY,
    params: SizingParams = SizingParams(),
) -> float:
    """Units to hold for one position. Returns 0.0 when inputs are unusable."""
    if equity <= 0 or price <= 0 or not math.isfinite(price):
        return 0.0
    if ann_vol <= 0 or not math.isfinite(ann_vol):
        return 0.0

    clamped = min(max(ann_vol, params.min_vol), params.max_vol)
    notional = equity * params.target_vol / clamped
    notional = min(notional, equity * params.max_position_pct)

    if notional <= 0:
        return 0.0

    raw = notional / price
    if asset_class is AssetClass.CRYPTO:
        return round(raw, params.crypto_precision)
    return float(math.floor(raw))


def position_risk_fraction(quantity: float, price: float, ann_vol: float, equity: float) -> float:
    """Share of equity this position puts at risk per year, for reporting."""
    if equity <= 0:
        return 0.0
    return (quantity * price * ann_vol) / equity

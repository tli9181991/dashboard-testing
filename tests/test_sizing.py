import math

import pytest

from sizing import SizingParams, annualized_vol_from_atr, target_quantity
from strategy import AssetClass


def test_vol_estimate_annualises_atr():
    assert annualized_vol_from_atr(2.0, 100.0) == pytest.approx(0.02 * math.sqrt(252))


def test_vol_estimate_rejects_bad_input():
    assert annualized_vol_from_atr(0.0, 100.0) == 0.0
    assert annualized_vol_from_atr(2.0, 0.0) == 0.0
    assert annualized_vol_from_atr(float("nan"), 100.0) == 0.0


def test_size_falls_as_volatility_rises():
    """The whole point: a noisier name gets a smaller position."""
    params = SizingParams(target_vol=0.15, max_position_pct=1.0)
    quiet = target_quantity(100_000, 50.0, 0.20, AssetClass.EQUITY, params)
    noisy = target_quantity(100_000, 50.0, 0.80, AssetClass.EQUITY, params)
    assert quiet > noisy
    assert noisy == pytest.approx(quiet / 4, rel=0.02)


def test_risk_contribution_is_equalised_across_names():
    """Two names at different prices and vols should carry the same vol budget."""
    params = SizingParams(target_vol=0.10, max_position_pct=1.0)
    a_qty = target_quantity(100_000, 30.0, 0.25, AssetClass.EQUITY, params)
    b_qty = target_quantity(100_000, 300.0, 0.50, AssetClass.EQUITY, params)
    a_risk = a_qty * 30.0 * 0.25
    b_risk = b_qty * 300.0 * 0.50
    assert a_risk == pytest.approx(b_risk, rel=0.01)
    assert a_risk == pytest.approx(100_000 * 0.10, rel=0.01)


def test_position_cap_binds_for_very_quiet_names():
    params = SizingParams(target_vol=0.15, max_position_pct=0.25, min_vol=0.01)
    qty = target_quantity(100_000, 10.0, 0.02, AssetClass.EQUITY, params)
    assert qty * 10.0 <= 100_000 * 0.25 + 1e-6


def test_vol_floor_prevents_unbounded_size():
    params = SizingParams(target_vol=0.15, max_position_pct=1.0, min_vol=0.05)
    at_floor = target_quantity(100_000, 10.0, 0.05, AssetClass.EQUITY, params)
    below = target_quantity(100_000, 10.0, 0.0001, AssetClass.EQUITY, params)
    assert below == at_floor


def test_equities_are_whole_shares_and_crypto_is_fractional():
    params = SizingParams()
    shares = target_quantity(100_000, 333.33, 0.30, AssetClass.EQUITY, params)
    coins = target_quantity(100_000, 3333.33, 0.60, AssetClass.CRYPTO, params)
    assert shares == float(int(shares))
    assert coins != float(int(coins))


def test_degenerate_inputs_return_zero():
    params = SizingParams()
    assert target_quantity(0, 100.0, 0.3, AssetClass.EQUITY, params) == 0.0
    assert target_quantity(100_000, 0.0, 0.3, AssetClass.EQUITY, params) == 0.0
    assert target_quantity(100_000, 100.0, 0.0, AssetClass.EQUITY, params) == 0.0


def test_fixed_share_count_would_not_equalise_risk():
    """Contrast with the old stock_share_size=6 behaviour."""
    old_a_risk = 6 * 30.0 * 0.25
    old_b_risk = 6 * 300.0 * 0.50
    assert old_b_risk / old_a_risk == pytest.approx(20.0)

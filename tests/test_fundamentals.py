"""Fundamentals: a missing field must stay missing."""

import pytest

import fundamentals as fnd


@pytest.fixture
def info():
    return {
        "longName": "Test Corp", "sector": "Technology", "industry": "Software",
        "currentPrice": 100.0, "trailingPE": 25.0, "profitMargins": 0.20,
        "marketCap": 5.0e11, "totalDebt": 1.0e10, "debtToEquity": 55.5,
        "revenueGrowth": 0.15, "recommendationKey": "buy",
        "numberOfAnalystOpinions": 30, "targetMeanPrice": 120.0, "beta": 1.1,
    }


def test_missing_fields_stay_none_not_zero(info):
    """A blank price-to-book and a price-to-book of zero mean different things."""
    snap = fnd.from_info("TEST", info)
    assert snap.get("Valuation", "Price / book") is None
    assert snap.get("Valuation", "Trailing P/E") == 25.0


def test_missing_values_render_as_a_dash():
    assert fnd.format_value(None, "pct") == "—"
    assert fnd.format_value(None, "money") == "—"


def test_non_numeric_and_nan_read_as_missing():
    snap = fnd.from_info("TEST", {"trailingPE": "n/a", "forwardPE": float("nan"),
                                  "priceToBook": float("inf"), "pegRatio": True})
    for label in ("Trailing P/E", "Forward P/E", "Price / book", "PEG"):
        assert snap.get("Valuation", label) is None, label


@pytest.mark.parametrize("value,kind,expected", [
    (0.223, "pct", "22.3%"),
    (1.42e12, "money", "$1.42T"),
    (9.8e9, "money", "$9.80B"),
    (5.0e6, "money", "$5.00M"),
    (123.456, "price", "$123.46"),
    (2.5e6, "count", "2.5M"),
    (28.4, "ratio", "28.40"),
])
def test_formatting(value, kind, expected):
    assert fnd.format_value(value, kind) == expected


def test_coverage_reports_what_the_vendor_returned(info):
    snap = fnd.from_info("TEST", info)
    assert 0.0 < snap.coverage < 1.0
    assert fnd.from_info("X", {"marketCap": 1e9}).coverage < 0.1


def test_upside_is_relative_to_the_last_price(info):
    snap = fnd.from_info("TEST", info)
    assert snap.upside() == pytest.approx(0.20)


def test_upside_is_none_without_a_target_or_price(info):
    no_target = fnd.from_info("TEST", {k: v for k, v in info.items()
                                       if k != "targetMeanPrice"})
    assert no_target.upside() is None
    assert fnd.from_info("TEST", {"targetMeanPrice": 120.0}).upside() is None


def test_prompt_text_omits_fields_the_vendor_did_not_supply(info):
    text = fnd.from_info("TEST", info).to_prompt_text()
    assert "Trailing P/E 25.00" in text
    assert "Price / book" not in text
    assert "—" not in text, "placeholders belong on screen, not in the model's context"


def test_prompt_text_states_the_coverage(info):
    assert "of the tracked fields" in fnd.from_info("TEST", info).to_prompt_text()


def test_prompt_text_says_so_when_there_is_nothing(info):
    snap = fnd.from_info("X", {})
    assert not snap.ok
    assert "unavailable" in snap.to_prompt_text()


def test_empty_info_is_not_ok():
    assert fnd.from_info("X", {}).ok is False
    assert fnd.from_info("X", {}).error


def test_rows_are_label_value_pairs_for_the_ui(info):
    rows = fnd.from_info("TEST", info).rows("Valuation")
    assert ("Trailing P/E", "25.00") in rows
    assert ("Price / book", "—") in rows


def test_an_explicit_price_overrides_the_info_field(info):
    snap = fnd.from_info("TEST", info, price=200.0)
    assert snap.price == 200.0
    assert snap.upside() == pytest.approx(-0.4)


def test_fetch_never_raises_on_a_bad_symbol():
    snap = fnd.fetch("__definitely_not_a_ticker__")
    assert isinstance(snap, fnd.FundamentalSnapshot)
    assert snap.ok is False or snap.coverage == 0.0

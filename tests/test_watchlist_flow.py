"""The screener -> watchlist -> monitoring path, without Streamlit.

Mirrors the store calls the tabs make so the wiring is covered even though the
Streamlit widgets themselves are not exercised here.
"""

import pytest

from positions import Holding
from strategy import AssetClass, Position
from watchlist import WatchlistStore


@pytest.fixture
def store(tmp_path):
    return WatchlistStore(tmp_path / "watchlist.json")


def _holding(symbol, quantity=5.0, avg_price=100.0):
    return Holding(
        symbol=symbol,
        position=Position(quantity=quantity, avg_price=avg_price),
        asset_class=AssetClass.infer(symbol),
    )


def _build_monitored(holdings, store):
    """The dict app.py builds at the top of each run."""
    watched = [s for s in store.symbols() if s not in holdings]
    monitored = {
        symbol: {"position": h.position, "asset_class": h.asset_class, "held": True}
        for symbol, h in holdings.items()
    }
    for symbol in watched:
        monitored[symbol] = {
            "position": Position(),
            "asset_class": AssetClass.infer(symbol),
            "held": False,
        }
    return monitored


def test_screener_pick_shows_up_in_monitoring(store):
    holdings = {"MRVL": _holding("MRVL")}
    assert list(_build_monitored(holdings, store)) == ["MRVL"]

    store.add("NVDA", sector="Technology", source="screener")

    monitored = _build_monitored(holdings, store)
    assert set(monitored) == {"MRVL", "NVDA"}
    assert monitored["NVDA"]["held"] is False
    assert monitored["MRVL"]["held"] is True


def test_watchlist_entries_are_evaluated_flat(store):
    """No position means the strategy reports an entry signal, not an exit."""
    store.add("NVDA")
    monitored = _build_monitored({}, store)
    position = monitored["NVDA"]["position"]
    assert position.quantity == 0
    assert not position.is_open
    assert position.unrealized_pnl(500.0) == 0.0


def test_a_held_symbol_is_never_duplicated_as_a_watch(store):
    """portfolio.csv wins: the name appears once, as a holding."""
    store.add("MRVL")
    holdings = {"MRVL": _holding("MRVL")}
    monitored = _build_monitored(holdings, store)
    assert list(monitored) == ["MRVL"]
    assert monitored["MRVL"]["held"] is True


def test_removing_from_the_watchlist_drops_it_from_monitoring(store):
    store.add("NVDA")
    store.add("TSM")
    assert set(_build_monitored({}, store)) == {"NVDA", "TSM"}

    store.remove("NVDA")
    assert set(_build_monitored({}, store)) == {"TSM"}


def test_bulk_removal_from_the_manage_expander(store):
    for symbol in ("NVDA", "TSM", "AMD"):
        store.add(symbol)
    assert store.remove_many(["NVDA", "AMD"]) == 2
    assert set(_build_monitored({}, store)) == {"TSM"}


def test_picks_survive_a_restart(store, tmp_path):
    store.add("NVDA", sector="Technology", source="screener")
    reopened = WatchlistStore(tmp_path / "watchlist.json")
    assert set(_build_monitored({}, reopened)) == {"NVDA"}


def test_crypto_symbols_keep_their_asset_class(store):
    store.add("ETH-USD")
    monitored = _build_monitored({}, store)
    assert monitored["ETH-USD"]["asset_class"] is AssetClass.CRYPTO


def test_nothing_to_monitor_when_both_sources_are_empty(store):
    assert _build_monitored({}, store) == {}


def test_button_state_reflects_membership(store):
    """Drives the three button states the screener renders."""
    holdings = {"MRVL": _holding("MRVL")}
    watched = set(store.symbols())

    assert "MRVL" in holdings                       # -> disabled "already a holding"
    assert "NVDA" not in holdings and "NVDA" not in watched   # -> "add"

    store.add("NVDA")
    watched = set(store.symbols())
    assert "NVDA" in watched                        # -> "watching (click to remove)"

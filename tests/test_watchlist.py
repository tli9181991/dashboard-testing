"""Watchlist persistence and de-duplication."""

import json

import pytest

from watchlist import WatchlistStore, normalize


@pytest.fixture
def store(tmp_path):
    return WatchlistStore(tmp_path / "watchlist.json")


def test_missing_file_reads_as_empty(store):
    assert store.entries() == []
    assert store.symbols() == []
    assert len(store) == 0


def test_added_symbols_survive_a_restart(store, tmp_path):
    assert store.add("NVDA") is True
    assert store.add("tsm") is True

    reopened = WatchlistStore(tmp_path / "watchlist.json")
    assert reopened.symbols() == ["NVDA", "TSM"]


def test_symbols_are_normalised(store):
    store.add("  msft \n")
    assert store.symbols() == ["MSFT"]
    assert store.contains("msft")
    assert store.contains("MSFT")


def test_adding_twice_is_a_no_op(store):
    assert store.add("AAPL") is True
    assert store.add("aapl") is False
    assert store.symbols() == ["AAPL"]


def test_blank_symbols_are_rejected(store):
    assert store.add("   ") is False
    assert store.add("") is False
    assert store.symbols() == []


def test_addition_order_is_preserved(store):
    for symbol in ("ZTS", "AAPL", "MRVL"):
        store.add(symbol)
    assert store.symbols() == ["ZTS", "AAPL", "MRVL"]


def test_remove_reports_whether_it_did_anything(store):
    store.add("NFLX")
    assert store.remove("nflx") is True
    assert store.remove("NFLX") is False
    assert store.symbols() == []


def test_remove_many_returns_a_count(store):
    for symbol in ("A", "B", "C", "D"):
        store.add(symbol)
    assert store.remove_many(["b", "d", "missing"]) == 2
    assert store.symbols() == ["A", "C"]


def test_clear_empties_the_list(store):
    store.add("AAPL")
    store.clear()
    assert store.symbols() == []
    assert store.path.exists(), "clearing should leave a valid empty file"


def test_metadata_is_kept(store):
    store.add("NVDA", sector="Technology", source="screener")
    entry = store.entries()[0]
    assert entry.sector == "Technology"
    assert entry.source == "screener"
    assert entry.added_at


def test_a_corrupt_file_reads_as_empty_rather_than_raising(store):
    store.path.write_text("{ not json")
    assert store.entries() == []
    # And it recovers on the next write.
    assert store.add("AAPL") is True
    assert store.symbols() == ["AAPL"]


def test_duplicates_already_on_disk_are_collapsed(store):
    store.path.write_text(json.dumps({"entries": [
        {"symbol": "AAPL", "added_at": "x"},
        {"symbol": "aapl", "added_at": "y"},
        {"symbol": "MSFT", "added_at": "z"},
    ]}))
    assert store.symbols() == ["AAPL", "MSFT"]


def test_a_bare_list_on_disk_is_accepted(store):
    store.path.write_text(json.dumps([{"symbol": "AAPL", "added_at": "x"}]))
    assert store.symbols() == ["AAPL"]


def test_malformed_rows_are_skipped(store):
    store.path.write_text(json.dumps({"entries": [
        {"symbol": "AAPL", "added_at": "x"},
        "not a dict",
        {"no_symbol": True},
        {"symbol": "   "},
        {"symbol": "MSFT", "added_at": "y"},
    ]}))
    assert store.symbols() == ["AAPL", "MSFT"]


def test_writes_leave_no_temp_files(store):
    store.add("AAPL")
    assert list(store.path.parent.glob("*.tmp")) == []


def test_parent_directory_is_created(tmp_path):
    store = WatchlistStore(tmp_path / "nested" / "deep" / "watchlist.json")
    assert store.add("AAPL") is True
    assert store.symbols() == ["AAPL"]


def test_normalize_helper():
    assert normalize("  eth-usd ") == "ETH-USD"

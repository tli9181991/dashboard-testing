"""Persistent watchlist of symbols to monitor without holding.

The monitoring tab reads two sources: ``portfolio.csv`` for positions you actually
own, and this file for names you want signals on but have no position in. A
watchlist entry is evaluated with a flat position, so the strategy reports the
entry signal rather than an exit.

Stored as JSON, written through a temp file and ``os.replace`` so an interrupted
save cannot truncate the list.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_PATH = Path("./watchlist.json")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def normalize(symbol: str) -> str:
    return str(symbol).strip().upper()


@dataclass(frozen=True)
class WatchlistEntry:
    symbol: str
    added_at: str
    sector: str = ""
    source: str = ""

    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "added_at": self.added_at,
                "sector": self.sector, "source": self.source}

    @classmethod
    def from_dict(cls, raw: dict) -> Optional["WatchlistEntry"]:
        symbol = normalize(raw.get("symbol", ""))
        if not symbol:
            return None
        return cls(
            symbol=symbol,
            added_at=str(raw.get("added_at", "")),
            sector=str(raw.get("sector", "")),
            source=str(raw.get("source", "")),
        )


class WatchlistStore:
    """Reads and writes the watchlist file. Order of addition is preserved."""

    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = Path(path)

    # ---- io ---------------------------------------------------------------
    def entries(self) -> list[WatchlistEntry]:
        """Stored entries. A missing or unreadable file reads as empty."""
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

        rows = raw.get("entries", []) if isinstance(raw, dict) else raw
        if not isinstance(rows, list):
            return []

        seen, out = set(), []
        for row in rows:
            if not isinstance(row, dict):
                continue
            entry = WatchlistEntry.from_dict(row)
            if entry is None or entry.symbol in seen:
                continue
            seen.add(entry.symbol)
            out.append(entry)
        return out

    def _write(self, entries: Iterable[WatchlistEntry]) -> None:
        payload = {"updated_at": _now(), "entries": [e.to_dict() for e in entries]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    # ---- queries ----------------------------------------------------------
    def symbols(self) -> list[str]:
        return [e.symbol for e in self.entries()]

    def contains(self, symbol: str) -> bool:
        return normalize(symbol) in set(self.symbols())

    def __len__(self) -> int:
        return len(self.entries())

    # ---- mutations --------------------------------------------------------
    def add(self, symbol: str, sector: str = "", source: str = "") -> bool:
        """Add a symbol. Returns False when it was already present or blank."""
        symbol = normalize(symbol)
        if not symbol:
            return False

        entries = self.entries()
        if any(e.symbol == symbol for e in entries):
            return False

        entries.append(WatchlistEntry(symbol, _now(), sector, source))
        self._write(entries)
        return True

    def remove(self, symbol: str) -> bool:
        """Remove a symbol. Returns False when it was not there."""
        symbol = normalize(symbol)
        entries = self.entries()
        remaining = [e for e in entries if e.symbol != symbol]
        if len(remaining) == len(entries):
            return False
        self._write(remaining)
        return True

    def remove_many(self, symbols: Iterable[str]) -> int:
        targets = {normalize(s) for s in symbols}
        entries = self.entries()
        remaining = [e for e in entries if e.symbol not in targets]
        removed = len(entries) - len(remaining)
        if removed:
            self._write(remaining)
        return removed

    def clear(self) -> None:
        self._write([])

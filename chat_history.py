"""On-disk chat history for the AI assistant tab.

Streamlit keeps ``st.session_state`` in memory only, so every conversation used to
disappear when the server restarted or the browser tab was closed. Conversations
are now persisted as one JSON file each, with a pointer to whichever is current so
the app reopens where it left off.

Layout::

    .chat_history/
        index.json                 {"current_id": "..."}
        conv_20260903-094619-a1b2c3.json

Writes go through a temp file and ``os.replace``, so an interrupted save leaves the
previous version intact rather than a half-written file.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

DEFAULT_DIR = Path("./.chat_history")
INDEX_FILE = "index.json"
CONVERSATION_PREFIX = "conv_"
TITLE_LENGTH = 48


def _now() -> str:
    # Microsecond resolution: conversations created within the same second must
    # still sort deterministically in the picker.
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def make_message(role: str, content: str) -> dict:
    """A single chat turn. ``role`` is 'user' or 'assistant'."""
    return {"role": role, "content": content, "timestamp": _now()}


@dataclass
class Conversation:
    id: str
    created_at: str
    updated_at: str
    messages: list[dict] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.messages

    @property
    def title(self) -> str:
        """First thing the user asked, trimmed — used to label the picker."""
        for message in self.messages:
            if message.get("role") == "user":
                text = " ".join(str(message.get("content", "")).split())
                if text:
                    return text[:TITLE_LENGTH] + ("…" if len(text) > TITLE_LENGTH else "")
        return "New chat"

    @property
    def started(self) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(self.created_at)
        except (TypeError, ValueError):
            return None

    def label(self) -> str:
        stamp = self.started
        when = stamp.strftime("%b %d, %H:%M") if stamp else "unknown"
        return f"{when} · {self.title} ({len(self.messages)})"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": self.messages,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Conversation":
        return cls(
            id=str(raw["id"]),
            created_at=str(raw.get("created_at", "")),
            updated_at=str(raw.get("updated_at", "")),
            messages=list(raw.get("messages", [])),
        )


class ChatHistoryStore:
    """Reads and writes conversations under a directory."""

    def __init__(self, directory: Path = DEFAULT_DIR):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    # ---- paths ------------------------------------------------------------
    @property
    def _index_path(self) -> Path:
        return self.directory / INDEX_FILE

    def _conversation_path(self, conversation_id: str) -> Path:
        return self.directory / f"{CONVERSATION_PREFIX}{conversation_id}.json"

    # ---- low-level io -----------------------------------------------------
    def _write_json(self, path: Path, payload: dict) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    def _read_json(self, path: Path) -> Optional[dict]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A corrupt or partially written file should not take the app down.
            return None

    # ---- index ------------------------------------------------------------
    def current_id(self) -> Optional[str]:
        index = self._read_json(self._index_path) or {}
        conversation_id = index.get("current_id")
        if conversation_id and self._conversation_path(conversation_id).exists():
            return conversation_id
        return None

    def set_current(self, conversation_id: str) -> None:
        self._write_json(self._index_path, {"current_id": conversation_id,
                                            "updated_at": _now()})

    # ---- conversations ----------------------------------------------------
    def create(self, make_current: bool = True) -> Conversation:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        conversation = Conversation(
            id=f"{stamp}-{uuid.uuid4().hex[:6]}",
            created_at=_now(),
            updated_at=_now(),
            messages=[],
        )
        self.save(conversation)
        if make_current:
            self.set_current(conversation.id)
        return conversation

    def save(self, conversation: Conversation) -> None:
        conversation.updated_at = _now()
        self._write_json(self._conversation_path(conversation.id), conversation.to_dict())

    def load(self, conversation_id: str) -> Optional[Conversation]:
        raw = self._read_json(self._conversation_path(conversation_id))
        if not raw or "id" not in raw:
            return None
        try:
            return Conversation.from_dict(raw)
        except (KeyError, TypeError):
            return None

    def delete(self, conversation_id: str) -> None:
        self._conversation_path(conversation_id).unlink(missing_ok=True)
        if self.current_id() == conversation_id:
            self._index_path.unlink(missing_ok=True)

    def list_conversations(self) -> list[Conversation]:
        """All stored conversations, newest first. Unreadable files are skipped."""
        found = []
        for path in self.directory.glob(f"{CONVERSATION_PREFIX}*.json"):
            raw = self._read_json(path)
            if not raw or "id" not in raw:
                continue
            try:
                found.append(Conversation.from_dict(raw))
            except (KeyError, TypeError):
                continue
        return sorted(found, key=lambda c: c.created_at, reverse=True)

    # ---- the call the app actually makes on startup -----------------------
    def load_current_or_create(self) -> Conversation:
        """Reopen the conversation last in use, else the newest, else start one."""
        conversation_id = self.current_id()
        if conversation_id:
            existing = self.load(conversation_id)
            if existing is not None:
                return existing

        newest = self.list_conversations()
        if newest:
            self.set_current(newest[0].id)
            return newest[0]

        return self.create()

    def start_new(self, current: Optional[Conversation] = None) -> Conversation:
        """Begin a fresh conversation, keeping the current one on disk.

        An untouched conversation is reused rather than leaving a trail of empty
        shells behind each time the button is pressed.
        """
        if current is not None and current.is_empty:
            self.set_current(current.id)
            return current
        return self.create()

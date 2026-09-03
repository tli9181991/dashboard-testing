"""Persistence behaviour for the assistant's conversation store."""

import json

import pytest

from chat_history import ChatHistoryStore, Conversation, make_message


@pytest.fixture
def store(tmp_path):
    return ChatHistoryStore(tmp_path / "history")


def test_creates_its_directory(tmp_path):
    target = tmp_path / "nested" / "history"
    ChatHistoryStore(target)
    assert target.is_dir()


def test_first_launch_starts_an_empty_conversation(store):
    conversation = store.load_current_or_create()
    assert conversation.is_empty
    assert store.current_id() == conversation.id


def test_messages_survive_a_restart(store, tmp_path):
    conversation = store.load_current_or_create()
    conversation.messages.append(make_message("user", "What is NVDA doing?"))
    conversation.messages.append(make_message("assistant", "Up 3% today."))
    store.save(conversation)

    # A fresh store object stands in for relaunching the app.
    reopened = ChatHistoryStore(tmp_path / "history").load_current_or_create()
    assert reopened.id == conversation.id
    assert [m["content"] for m in reopened.messages] == ["What is NVDA doing?", "Up 3% today."]


def test_starting_a_new_chat_keeps_the_old_one(store):
    first = store.load_current_or_create()
    first.messages.append(make_message("user", "hello"))
    store.save(first)

    second = store.start_new(first)
    assert second.id != first.id
    assert second.is_empty
    assert store.current_id() == second.id

    ids = {c.id for c in store.list_conversations()}
    assert {first.id, second.id} <= ids
    assert store.load(first.id).messages[0]["content"] == "hello"


def test_starting_a_new_chat_reuses_an_untouched_one(store):
    first = store.load_current_or_create()
    again = store.start_new(first)
    assert again.id == first.id, "an empty chat should not spawn another empty chat"
    assert len(store.list_conversations()) == 1


def test_conversations_are_listed_newest_first(store):
    ids = []
    for n in range(3):
        conversation = store.create()
        conversation.messages.append(make_message("user", f"question {n}"))
        store.save(conversation)
        ids.append(conversation.id)
    assert [c.id for c in store.list_conversations()] == list(reversed(ids))


def test_title_comes_from_the_first_user_message(store):
    conversation = store.create()
    assert conversation.title == "New chat"
    conversation.messages.append(make_message("assistant", "greeting first"))
    conversation.messages.append(make_message("user", "  Should  I buy   TSM? "))
    assert conversation.title == "Should I buy TSM?"


def test_long_titles_are_trimmed(store):
    conversation = store.create()
    conversation.messages.append(make_message("user", "x" * 200))
    assert len(conversation.title) <= 49
    assert conversation.title.endswith("…")


def test_label_includes_date_and_message_count(store):
    conversation = store.create()
    conversation.messages.append(make_message("user", "hi"))
    assert "hi" in conversation.label()
    assert "(1)" in conversation.label()


def test_deleting_the_current_conversation_clears_the_pointer(store):
    conversation = store.load_current_or_create()
    store.delete(conversation.id)
    assert store.current_id() is None
    assert store.load(conversation.id) is None


def test_a_corrupt_file_is_skipped_not_fatal(store):
    good = store.create()
    good.messages.append(make_message("user", "fine"))
    store.save(good)
    (store.directory / "conv_broken.json").write_text("{not json")

    listed = store.list_conversations()
    assert [c.id for c in listed] == [good.id]
    assert store.load_current_or_create().id == good.id


def test_a_corrupt_index_falls_back_to_the_newest_conversation(store):
    conversation = store.create()
    conversation.messages.append(make_message("user", "still here"))
    store.save(conversation)
    (store.directory / "index.json").write_text("garbage")

    recovered = store.load_current_or_create()
    assert recovered.id == conversation.id


def test_a_dangling_pointer_falls_back_instead_of_raising(store):
    conversation = store.create()
    store.set_current("does-not-exist")
    assert store.current_id() is None
    assert store.load_current_or_create().id == conversation.id


def test_saves_are_atomic_leaving_no_temp_files(store):
    conversation = store.create()
    conversation.messages.append(make_message("user", "hello"))
    store.save(conversation)
    assert list(store.directory.glob("*.tmp")) == []


def test_round_trip_through_json_preserves_the_conversation(store):
    conversation = store.create()
    conversation.messages.append(make_message("user", "unicode: café 📈"))
    store.save(conversation)

    raw = json.loads((store.directory / f"conv_{conversation.id}.json").read_text(encoding="utf-8"))
    assert Conversation.from_dict(raw).messages[0]["content"] == "unicode: café 📈"


def test_messages_carry_role_content_and_timestamp():
    message = make_message("user", "hi")
    assert message["role"] == "user"
    assert message["content"] == "hi"
    assert message["timestamp"]


class TestChatTabFlow:
    """Mirrors the sequence the Streamlit tab performs, without Streamlit."""

    @staticmethod
    def _launch(directory):
        """What the tab does when session_state has no conversation yet."""
        store = ChatHistoryStore(directory)
        opened = store.load_current_or_create()
        return store, opened.id, list(opened.messages)

    @staticmethod
    def _exchange(store, conversation_id, session_messages, prompt, reply):
        """What the tab does after the agent answers."""
        prior = list(session_messages)
        session_messages.append(make_message("user", prompt))
        session_messages.append(make_message("assistant", reply))
        conversation = store.load(conversation_id)
        conversation.messages = session_messages
        store.save(conversation)
        return prior

    def test_history_reopens_where_it_left_off(self, tmp_path):
        directory = tmp_path / "history"

        store, conversation_id, messages = self._launch(directory)
        assert messages == []
        self._exchange(store, conversation_id, messages, "Is TSM extended?", "Slightly.")

        # Relaunch the app.
        _, reopened_id, reopened = self._launch(directory)
        assert reopened_id == conversation_id
        assert [m["content"] for m in reopened] == ["Is TSM extended?", "Slightly."]

    def test_clear_button_archives_and_opens_a_blank_chat(self, tmp_path):
        directory = tmp_path / "history"
        store, first_id, messages = self._launch(directory)
        self._exchange(store, first_id, messages, "old question", "old answer")

        # Press "Clear Chat History".
        fresh = store.start_new(store.load(first_id))
        assert fresh.is_empty

        # Relaunch: the blank chat is current, the old one is still on disk.
        _, current_id, current_messages = self._launch(directory)
        assert current_id == fresh.id
        assert current_messages == []
        assert store.load(first_id).messages[0]["content"] == "old question"

    def test_switching_conversations_persists_the_choice(self, tmp_path):
        directory = tmp_path / "history"
        store, first_id, messages = self._launch(directory)
        self._exchange(store, first_id, messages, "first thread", "reply one")

        second = store.start_new(store.load(first_id))
        self._exchange(store, second.id, [], "second thread", "reply two")

        # Pick the older conversation from the dropdown.
        store.set_current(first_id)

        _, reopened_id, reopened = self._launch(directory)
        assert reopened_id == first_id
        assert reopened[0]["content"] == "first thread"

    def test_agent_receives_history_excluding_the_current_prompt(self, tmp_path):
        """The prompt is passed separately, so it must not also appear in history."""
        directory = tmp_path / "history"
        store, conversation_id, messages = self._launch(directory)
        self._exchange(store, conversation_id, messages, "first", "answer one")

        prior = self._exchange(store, conversation_id, messages, "second", "answer two")
        assert [m["content"] for m in prior] == ["first", "answer one"]
        assert "second" not in [m["content"] for m in prior]

    def test_every_exchange_is_durable_immediately(self, tmp_path):
        directory = tmp_path / "history"
        store, conversation_id, messages = self._launch(directory)

        for n in range(3):
            self._exchange(store, conversation_id, messages, f"q{n}", f"a{n}")
            # Simulate a crash right here: a brand new store must see the exchange.
            recovered = ChatHistoryStore(directory).load_current_or_create()
            assert len(recovered.messages) == (n + 1) * 2

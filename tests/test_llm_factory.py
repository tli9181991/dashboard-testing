"""One place builds the chat model, so one place is worth testing."""

import pytest

import llm as llm_factory


def test_no_key_means_no_credentials(monkeypatch):
    monkeypatch.setattr(llm_factory, "GOOGLE_API_KEY", "")
    assert llm_factory.credentials_present() is False


def test_a_key_means_credentials(monkeypatch):
    monkeypatch.setattr(llm_factory, "GOOGLE_API_KEY", "abc123")
    assert llm_factory.credentials_present() is True


def test_building_without_a_key_raises_rather_than_half_building(monkeypatch):
    """A model object that fails on first use is harder to diagnose than one
    that never existed."""
    monkeypatch.setattr(llm_factory, "GOOGLE_API_KEY", "")
    with pytest.raises(RuntimeError, match="Gemini API key"):
        llm_factory.get_chat_model()


def test_the_model_carries_the_configured_name(monkeypatch):
    monkeypatch.setattr(llm_factory, "GOOGLE_API_KEY", "abc123")
    monkeypatch.setattr(llm_factory, "GEMINI_MODEL_NAME", "gemini-3.5-flash")
    model = llm_factory.get_chat_model()
    assert "gemini-3.5-flash" in str(model.model)


def test_an_explicit_model_overrides_the_default(monkeypatch):
    monkeypatch.setattr(llm_factory, "GOOGLE_API_KEY", "abc123")
    model = llm_factory.get_chat_model(model="gemini-2.5-flash")
    assert "gemini-2.5-flash" in str(model.model)


def test_temperature_is_passed_through(monkeypatch):
    monkeypatch.setattr(llm_factory, "GOOGLE_API_KEY", "abc123")
    assert llm_factory.get_chat_model(temperature=0.7).temperature == 0.7


def test_either_key_name_is_accepted(monkeypatch):
    """Both spellings are in common use; ignoring the one the user set is a bad
    first five minutes."""
    import importlib

    import config

    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv(name, "from-" + name)
        importlib.reload(config)
        assert config.GOOGLE_API_KEY == "from-" + name, name

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-wins")
    importlib.reload(config)
    assert config.GOOGLE_API_KEY == "google-wins"


def test_the_model_name_is_configurable(monkeypatch):
    """The model id is the one thing likely to need changing, so it must not be
    hard-coded anywhere but config."""
    import importlib

    import config

    monkeypatch.setenv("GEMINI_MODEL_NAME", "gemini-9-turbo")
    importlib.reload(config)
    assert config.GEMINI_MODEL_NAME == "gemini-9-turbo"


def test_nothing_in_the_source_still_reaches_for_azure():
    import pathlib

    offenders = []
    for path in pathlib.Path(".").glob("*.py"):
        text = path.read_text()
        if "AZURE_INFERENCE" in text or "AzureAIChatCompletionsModel" in text:
            offenders.append(path.name)
    assert not offenders, offenders


def test_every_llm_call_site_goes_through_the_factory():
    """Before this the constructor was copy-pasted at four sites, which is how a
    provider migration turns into a scavenger hunt."""
    import pathlib
    import re

    direct = []
    for path in pathlib.Path(".").glob("*.py"):
        if path.name == "llm.py":
            continue
        if re.search(r"ChatGoogleGenerativeAI\s*\(", path.read_text()):
            direct.append(path.name)
    assert not direct, f"these build the model directly instead of via llm.py: {direct}"

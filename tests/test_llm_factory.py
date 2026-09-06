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
    monkeypatch.setattr(llm_factory, "GEMINI_MODEL_NAME", "gemini-3.6-flash")
    model = llm_factory.get_chat_model()
    assert "gemini-3.6-flash" in str(model.model)


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


# ---------------------------------------------------------------------------
# Model discovery — the authoritative answer to "is that model id real"
# ---------------------------------------------------------------------------

class _Response:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.setattr(llm_factory, "GOOGLE_API_KEY", "abc123")


def _patch_get(monkeypatch, response):
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: response)


def test_listing_without_a_key_reports_that(monkeypatch):
    monkeypatch.setattr(llm_factory, "GOOGLE_API_KEY", "")
    out = llm_factory.list_models()
    assert out["models"] == []
    assert "Gemini API key" in out["error"]


def test_only_models_that_can_generate_are_listed(keyed, monkeypatch):
    """An embedding model in the list would be a model id that never works."""
    _patch_get(monkeypatch, _Response(200, {"models": [
        {"name": "models/gemini-2.5-flash", "displayName": "Gemini 2.5 Flash",
         "supportedGenerationMethods": ["generateContent"], "inputTokenLimit": 1000000},
        {"name": "models/text-embedding-004", "displayName": "Embedding",
         "supportedGenerationMethods": ["embedContent"]},
    ]}))
    ids = [m["id"] for m in llm_factory.list_models()["models"]]
    assert ids == ["gemini-2.5-flash"]


def test_the_key_travels_in_a_header_not_the_url(keyed, monkeypatch):
    """A key in a query string ends up in proxy logs and shell history."""
    seen = {}

    import requests

    def capture(url, **kwargs):
        seen["url"] = url
        seen["headers"] = kwargs.get("headers", {})
        return _Response(200, {"models": []})

    monkeypatch.setattr(requests, "get", capture)
    llm_factory.list_models()
    assert "abc123" not in seen["url"]
    assert seen["headers"].get("x-goog-api-key") == "abc123"


def test_an_auth_failure_explains_itself(keyed, monkeypatch):
    _patch_get(monkeypatch, _Response(403, {"error": {"message": "API key not valid"}}))
    out = llm_factory.list_models()
    assert out["models"] == []
    assert "API key not valid" in out["error"]
    assert "Generative Language API is enabled" in out["error"]


def test_an_unreachable_api_is_reported_not_raised(keyed, monkeypatch):
    import requests

    def boom(*a, **k):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(requests, "get", boom)
    out = llm_factory.list_models()
    assert "Could not reach" in out["error"]


def test_a_configured_model_in_the_list_is_confirmed(keyed, monkeypatch):
    _patch_get(monkeypatch, _Response(200, {"models": [
        {"name": "models/gemini-2.5-flash",
         "supportedGenerationMethods": ["generateContent"]},
    ]}))
    text = llm_factory.render_models(llm_factory.list_models(),
                                     configured="gemini-2.5-flash")
    assert "is valid" in text


def test_a_configured_model_missing_from_the_list_is_called_out(keyed, monkeypatch):
    """This is the whole point: say plainly that the id is why calls fail, and
    name the alternatives rather than leaving the user to guess."""
    _patch_get(monkeypatch, _Response(200, {"models": [
        {"name": "models/gemini-2.5-flash",
         "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-2.5-pro",
         "supportedGenerationMethods": ["generateContent"]},
    ]}))
    text = llm_factory.render_models(llm_factory.list_models(),
                                     configured="gemini-3.6-flash")
    assert "is NOT in the list" in text
    assert "gemini-2.5-flash" in text
    assert "Set GEMINI_MODEL_NAME" in text


def test_preview_and_experimental_models_are_not_suggested_as_the_fix(keyed, monkeypatch):
    """Suggesting a preview id as the stable replacement invites a second failure."""
    _patch_get(monkeypatch, _Response(200, {"models": [
        {"name": "models/gemini-2.5-flash",
         "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-9-flash-preview",
         "supportedGenerationMethods": ["generateContent"]},
    ]}))
    text = llm_factory.render_models(llm_factory.list_models(), configured="nope")
    suggestion = text.split("Flash models available: ")[1].split("\n")[0]
    assert "gemini-2.5-flash" in suggestion
    assert "preview" not in suggestion


# ---------------------------------------------------------------------------
# Key doctor — "API key not valid" means a value was found and rejected
# ---------------------------------------------------------------------------

VALID_SHAPE = "AIza" + "b" * 35   # 39 chars, correct prefix


def _in_dir(monkeypatch, tmp_path, env_text=None):
    monkeypatch.chdir(tmp_path)
    if env_text is not None:
        (tmp_path / ".env").write_text(env_text)


def test_a_shell_export_silently_beating_dotenv_is_caught(monkeypatch, tmp_path):
    """load_dotenv does not override an exported variable, and nothing says so —
    the app then uses a stale key while .env looks correct."""
    _in_dir(monkeypatch, tmp_path, f"GOOGLE_API_KEY={VALID_SHAPE}\n")
    stale = "AIza" + "z" * 35
    monkeypatch.setenv("GOOGLE_API_KEY", stale)
    monkeypatch.setattr(llm_factory, "GOOGLE_API_KEY", stale)

    problems = " ".join(llm_factory.diagnose_key()["problems"])
    assert "does NOT match the one in .env" in problems
    assert "unset GOOGLE_API_KEY" in problems


def test_a_matching_key_is_not_reported_as_overridden(monkeypatch, tmp_path):
    _in_dir(monkeypatch, tmp_path, f"GOOGLE_API_KEY={VALID_SHAPE}\n")
    monkeypatch.setenv("GOOGLE_API_KEY", VALID_SHAPE)
    monkeypatch.setattr(llm_factory, "GOOGLE_API_KEY", VALID_SHAPE)
    assert llm_factory.diagnose_key()["problems"] == []


def test_quotes_stripped_by_dotenv_are_not_flagged(monkeypatch, tmp_path):
    """Regression: the raw file was inspected, so standard quoting — which
    python-dotenv strips — produced a false alarm."""
    _in_dir(monkeypatch, tmp_path, f'GOOGLE_API_KEY="{VALID_SHAPE}"\n')
    monkeypatch.setenv("GOOGLE_API_KEY", VALID_SHAPE)
    monkeypatch.setattr(llm_factory, "GOOGLE_API_KEY", VALID_SHAPE)
    assert llm_factory.diagnose_key()["problems"] == []


def test_quotes_that_survive_into_the_value_are_flagged(monkeypatch, tmp_path):
    _in_dir(monkeypatch, tmp_path, "")
    quoted = f'"{VALID_SHAPE}"'
    monkeypatch.setenv("GOOGLE_API_KEY", quoted)
    monkeypatch.setattr(llm_factory, "GOOGLE_API_KEY", quoted)
    assert any("wrapped in quotes" in p for p in llm_factory.diagnose_key()["problems"])


def test_a_wrong_prefix_suggests_it_is_not_an_api_key(monkeypatch, tmp_path):
    _in_dir(monkeypatch, tmp_path, "")
    monkeypatch.setenv("GOOGLE_API_KEY", "ya29." + "x" * 34)
    monkeypatch.setattr(llm_factory, "GOOGLE_API_KEY", "ya29." + "x" * 34)
    problems = " ".join(llm_factory.diagnose_key()["problems"])
    assert "does not start with 'AIza'" in problems
    assert "OAuth client id" in problems


def test_a_truncated_key_is_caught_by_length(monkeypatch, tmp_path):
    _in_dir(monkeypatch, tmp_path, "")
    monkeypatch.setenv("GOOGLE_API_KEY", "AIzaShort")
    monkeypatch.setattr(llm_factory, "GOOGLE_API_KEY", "AIzaShort")
    assert any("characters; Google API keys are 39" in p
               for p in llm_factory.diagnose_key()["problems"])


def test_whitespace_around_the_value_is_caught(monkeypatch, tmp_path):
    _in_dir(monkeypatch, tmp_path, "")
    monkeypatch.setenv("GOOGLE_API_KEY", VALID_SHAPE + " ")
    monkeypatch.setattr(llm_factory, "GOOGLE_API_KEY", VALID_SHAPE + " ")
    assert any("whitespace" in p for p in llm_factory.diagnose_key()["problems"])


def test_a_missing_env_file_is_reported(monkeypatch, tmp_path):
    _in_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(llm_factory, "GOOGLE_API_KEY", "")
    report = llm_factory.diagnose_key()
    assert any("no .env file found" in f for f in report["findings"])


def test_a_well_formed_key_says_the_loading_is_fine(monkeypatch, tmp_path):
    """So a genuinely revoked key is not mistaken for a config problem."""
    _in_dir(monkeypatch, tmp_path, f"GOOGLE_API_KEY={VALID_SHAPE}\n")
    monkeypatch.setenv("GOOGLE_API_KEY", VALID_SHAPE)
    monkeypatch.setattr(llm_factory, "GOOGLE_API_KEY", VALID_SHAPE)
    text = llm_factory.render_diagnosis(llm_factory.diagnose_key())
    assert "Nothing wrong" in text
    assert "invalid or revoked" in text


def test_the_diagnosis_never_prints_the_key(monkeypatch, tmp_path):
    """It reports shape — length, prefix, last four — and nothing more."""
    _in_dir(monkeypatch, tmp_path, f"GOOGLE_API_KEY={VALID_SHAPE}\n")
    monkeypatch.setenv("GOOGLE_API_KEY", VALID_SHAPE)
    monkeypatch.setattr(llm_factory, "GOOGLE_API_KEY", VALID_SHAPE)
    text = llm_factory.render_diagnosis(llm_factory.diagnose_key())
    assert VALID_SHAPE not in text
    assert VALID_SHAPE[:20] not in text


# ---------------------------------------------------------------------------
# 403s that look alike and need different fixes
# ---------------------------------------------------------------------------

BLOCKED = ("Requests to this API generativelanguage.googleapis.com method "
           "google.ai.generativelanguage.v1beta.ModelService.ListModels are blocked.")
NOT_ENABLED = ("Generative Language API has not been used in project 12345 before "
               "or it is disabled.")


def test_a_key_restriction_is_named_as_such(keyed, monkeypatch):
    """The key is valid; its allowed-API list just excludes this API. Saying
    "check the key is valid" would send the user to rotate a working key."""
    _patch_get(monkeypatch, _Response(403, {"error": {"message": BLOCKED}}))
    error = llm_factory.list_models()["error"]
    assert "the key is fine" in error
    assert "API restrictions" in error
    assert "Don't restrict key" in error


def test_a_disabled_api_gets_the_other_fix(keyed, monkeypatch):
    _patch_get(monkeypatch, _Response(403, {"error": {"message": NOT_ENABLED}}))
    error = llm_factory.list_models()["error"]
    assert "not enabled for this key's project" in error
    assert "Library" in error


def test_the_two_403s_do_not_get_the_same_advice(keyed, monkeypatch):
    _patch_get(monkeypatch, _Response(403, {"error": {"message": BLOCKED}}))
    blocked = llm_factory.list_models()["error"]
    _patch_get(monkeypatch, _Response(403, {"error": {"message": NOT_ENABLED}}))
    disabled = llm_factory.list_models()["error"]
    assert blocked != disabled


def test_an_unrecognised_403_still_gets_generic_advice(keyed, monkeypatch):
    _patch_get(monkeypatch, _Response(403, {"error": {"message": "something else"}}))
    assert "Check the key is valid" in llm_factory.list_models()["error"]


def test_an_invalid_key_is_not_treated_as_a_restriction(keyed, monkeypatch):
    """A 400 on the key value must not send the user into the restrictions page."""
    _patch_get(monkeypatch, _Response(400, {"error": {"message": "API key not valid"}}))
    error = llm_factory.list_models()["error"]
    assert "API key not valid" in error
    assert "API restrictions" not in error

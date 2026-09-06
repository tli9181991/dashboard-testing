"""The one place the app builds a chat model.

Every LLM feature — the assistant agent, news sentiment, sector commentary — goes
through ``get_chat_model`` so the provider, model name and credential are decided
once. Before this the constructor was copy-pasted at four call sites, which is how
a provider migration turns into a scavenger hunt.

Credentials come from ``GOOGLE_API_KEY``, falling back to ``GEMINI_API_KEY``:
``langchain-google-genai`` reads the first from the environment itself, but both
names are in common use and silently ignoring the one the user actually set is a
bad first five minutes.
"""

from __future__ import annotations

from config import GEMINI_MODEL_NAME, GOOGLE_API_KEY

MISSING_CREDENTIALS = (
    "Missing Gemini API key. Set GOOGLE_API_KEY (or GEMINI_API_KEY) in your .env file."
)


def credentials_present() -> bool:
    return bool(GOOGLE_API_KEY)


def get_chat_model(temperature: float = 0.0, model: str | None = None):
    """A configured Gemini chat model, or raise if there is no key.

    Callers that surface a friendly message should check ``credentials_present``
    first; this raises rather than returning a half-built client, because a model
    object that fails on first use is harder to diagnose than one that never
    existed.
    """
    if not credentials_present():
        raise RuntimeError(MISSING_CREDENTIALS)

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model or GEMINI_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
    )

MODELS_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"


def list_models(timeout: int = 20) -> dict:
    """Ask Google which models this key can actually call.

    The authoritative answer to "is that model id real". Model names move faster
    than any list written into a repo, so this asks rather than assumes.

    Returns ``{"models": [...], "error": ""}``; ``models`` holds only the ids that
    support ``generateContent``, which is the method every feature here uses.
    """
    if not credentials_present():
        return {"models": [], "error": MISSING_CREDENTIALS}

    import requests

    try:
        # The key goes in a header, not the query string, so it stays out of
        # proxy logs and shell history.
        response = requests.get(
            MODELS_ENDPOINT,
            headers={"x-goog-api-key": GOOGLE_API_KEY},
            timeout=timeout,
        )
    except Exception as exc:
        return {"models": [], "error": f"Could not reach the Gemini API: {exc}"}

    if response.status_code != 200:
        detail = ""
        try:
            detail = response.json().get("error", {}).get("message", "")
        except Exception:
            detail = response.text[:200]
        hint = _http_hint(response.status_code, detail)
        return {"models": [],
                "error": f"HTTP {response.status_code} listing models. {detail}{hint}"}

    usable = []
    for entry in response.json().get("models", []):
        if "generateContent" not in entry.get("supportedGenerationMethods", []):
            continue
        usable.append({
            "id": entry.get("name", "").removeprefix("models/"),
            "display_name": entry.get("displayName", ""),
            "input_token_limit": entry.get("inputTokenLimit"),
        })
    return {"models": sorted(usable, key=lambda m: m["id"]), "error": ""}


def _http_hint(status_code: int, detail: str) -> str:
    """Turn Google's error into the specific console fix.

    The two 403s look alike and have different remedies: a key whose *API
    restriction* list excludes this API, versus an API that was never *enabled*
    for the project. Google words them differently, so they can be told apart.
    """
    lowered = (detail or "").lower()

    if "are blocked" in lowered or "api_key_service_blocked" in lowered:
        return (
            "\n\nThis is the API key's own restriction list, not the key's validity — "
            "the key is fine, it is just not allowed to call this API.\n"
            "Fix it in the Google Cloud console:\n"
            "  1. APIs & Services > Library > enable 'Generative Language API'\n"
            "     for this key's project. It must be enabled before it can be\n"
            "     selected in step 2.\n"
            "  2. APIs & Services > Credentials > your key > API restrictions:\n"
            "     add 'Generative Language API' to the allowed list, or choose\n"
            "     'Don't restrict key' to confirm the diagnosis quickly.\n"
            "Restrictions can take a minute or two to propagate."
        )

    if "has not been used" in lowered or "is disabled" in lowered:
        return (
            "\n\nThe Generative Language API is not enabled for this key's project.\n"
            "Enable it at APIs & Services > Library, then retry."
        )

    if status_code in (401, 403):
        return (" Check the key is valid and that the Generative Language API is "
                "enabled for its project.")
    return ""


def render_models(payload: dict, configured: str | None = None) -> str:
    """The model list as text, flagging whether the configured id is in it."""
    if payload["error"]:
        return payload["error"]
    if not payload["models"]:
        return "The API returned no models supporting generateContent for this key."

    configured = configured or GEMINI_MODEL_NAME
    ids = [m["id"] for m in payload["models"]]
    lines = [f"{len(ids)} models this key can call:"]
    lines += [f"  {m['id']}" + (f"  ({m['display_name']})" if m["display_name"] else "")
              for m in payload["models"]]

    lines.append("")
    if configured in ids:
        lines.append(f"GEMINI_MODEL_NAME={configured} is valid.")
    else:
        flash = [i for i in ids if "flash" in i and "preview" not in i and "exp" not in i]
        lines.append(f"GEMINI_MODEL_NAME={configured} is NOT in the list — this is why "
                     "calls fail.")
        if flash:
            lines.append("Flash models available: " + ", ".join(flash))
        lines.append("Set GEMINI_MODEL_NAME in .env to one of the ids above.")
    return "\n".join(lines)


#: Google API keys are 39 characters and begin with this constant prefix.
KEY_PREFIX = "AIza"
KEY_LENGTH = 39


def diagnose_key() -> dict:
    """Why the key the app loaded might not be the key you think it is.

    "API key not valid" means a value *was* found and Google rejected it, so the
    useful questions are where it came from and what shape it is. Two traps this
    catches:

    * ``load_dotenv()`` does not override variables already exported in the shell,
      so a stale export silently beats the .env file and nothing says so.
    * A value copied with surrounding quotes or a trailing newline reaches the API
      as a different string than the one on screen.

    Reports shape only — length, prefix, last four — never the key.
    """
    import os
    from pathlib import Path

    findings: list[str] = []
    problems: list[str] = []

    env_path = None
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"):
        if candidate.exists():
            env_path = candidate
            break

    file_values: dict[str, str] = {}
    if env_path:
        findings.append(f".env found at {env_path}")
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, raw = line.partition("=")
            name = name.strip()
            if name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
                file_values[name] = raw
    else:
        findings.append("no .env file found in the working directory")
        problems.append("Create .env at the repo root, or export the key in your shell.")

    loaded = GOOGLE_API_KEY
    if not loaded:
        findings.append("no key loaded")
        problems.append("Neither GOOGLE_API_KEY nor GEMINI_API_KEY resolved to a value.")
        return {"findings": findings, "problems": problems, "env_path": str(env_path or "")}

    source = "GOOGLE_API_KEY" if os.environ.get("GOOGLE_API_KEY") else "GEMINI_API_KEY"
    findings.append(f"key loaded from {source}: {len(loaded)} chars, ends …{loaded[-4:]}")

    # The shell wins over .env, and that is exactly the silent failure.
    for name, raw in file_values.items():
        in_file = raw.strip().strip("\"'")
        if in_file and in_file != loaded and os.environ.get(name):
            problems.append(
                f"The {name} the app is using does NOT match the one in .env. "
                "load_dotenv() does not override a variable already exported in "
                f"your shell, so the shell's stale value wins. Run: unset {name}"
            )

    if not loaded.startswith(KEY_PREFIX):
        problems.append(
            f"The key does not start with '{KEY_PREFIX}'. Google API keys do. "
            "This may be an OAuth client id, a service-account credential, or a "
            "truncated paste rather than an API key."
        )
    if len(loaded) != KEY_LENGTH:
        problems.append(
            f"The key is {len(loaded)} characters; Google API keys are {KEY_LENGTH}. "
            "It looks truncated or to have picked up extra characters."
        )
    if loaded != loaded.strip():
        problems.append("The loaded key has leading or trailing whitespace — check "
                        "for a stray space or newline after the = in .env.")
    if len(loaded) >= 2 and loaded[0] == loaded[-1] and loaded[0] in "\"'":
        problems.append("The loaded key is still wrapped in quotes. python-dotenv "
                        "strips standard quoting, so these came through some other "
                        "way — remove them from .env.")

    return {"findings": findings, "problems": problems, "env_path": str(env_path or "")}


def render_diagnosis(report: dict) -> str:
    lines = ["Key diagnosis:"]
    lines += [f"  {f}" for f in report["findings"]]
    if report["problems"]:
        lines.append("")
        lines.append("Problems found:")
        lines += [f"  - {p}" for p in report["problems"]]
    else:
        lines.append("")
        lines.append("  Nothing wrong with how the key is being loaded or its shape.")
        lines.append("  If Google still rejects it, the key itself is invalid or "
                     "revoked — rotate it in the console, or check the Generative "
                     "Language API is enabled for its project.")
    return "\n".join(lines)


if __name__ == "__main__":
    payload = list_models()
    print(render_models(payload))
    # A rejected key is about the key, not the model list — say why.
    if payload["error"]:
        print()
        print(render_diagnosis(diagnose_key()))

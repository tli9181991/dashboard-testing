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
        hint = ""
        if response.status_code in (401, 403):
            hint = (" Check the key is valid and that the Generative Language "
                    "API is enabled for its project.")
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


if __name__ == "__main__":
    print(render_models(list_models()))

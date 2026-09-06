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

"""The 2-day news window: stale headlines must not be scored as today's."""

from datetime import datetime, timedelta, timezone

import pytest

import sentiment


def _story(hours_ago, title="Headline", publisher="Wire"):
    when = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return {"content": {"title": title, "pubDate": when.isoformat(),
                        "provider": {"displayName": publisher},
                        "summary": "body"}}


@pytest.fixture
def patched(monkeypatch):
    def install(stories):
        class FakeTicker:
            def __init__(self, symbol): self.news = stories
        monkeypatch.setattr(sentiment.yf, "Ticker", FakeTicker)
    return install


def test_epoch_seconds_are_understood():
    epoch = datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp()
    assert sentiment._published_at({"providerPublishTime": epoch}).year == 2026


def test_iso_strings_are_understood():
    got = sentiment._published_at({"content": {"pubDate": "2026-03-01T10:00:00Z"}})
    assert got.year == 2026 and got.tzinfo is not None


def test_a_missing_timestamp_reads_as_unknown():
    assert sentiment._published_at({"content": {"title": "x"}}) is None


def test_unparseable_timestamps_read_as_unknown():
    assert sentiment._published_at({"content": {"pubDate": "last tuesday"}}) is None


def test_only_stories_inside_the_window_are_kept(patched):
    patched([_story(2), _story(20), _story(70), _story(400)])
    news = sentiment.collect_recent_news("AAA", days=2)
    assert len(news["articles"]) == 2
    assert news["stale"] == 2
    assert news["fetched"] == 4


def test_a_wider_window_keeps_more(patched):
    patched([_story(2), _story(20), _story(70), _story(400)])
    assert len(sentiment.collect_recent_news("AAA", days=7)["articles"]) == 3


def test_undated_stories_are_dropped_rather_than_assumed_recent(patched):
    patched([{"content": {"title": "no date"}}, _story(1)])
    news = sentiment.collect_recent_news("AAA", days=2)
    assert len(news["articles"]) == 1
    assert news["stale"] == 1


def test_articles_come_back_newest_first(patched):
    patched([_story(40, "older"), _story(1, "newest"), _story(20, "middle")])
    titles = [a["title"] for a in sentiment.collect_recent_news("AAA", days=2)["articles"]]
    assert titles == ["newest", "middle", "older"]


def test_the_item_cap_is_respected(patched):
    patched([_story(h, f"story {h}") for h in range(1, 30)])
    assert len(sentiment.collect_recent_news("AAA", days=2, max_items=5)["articles"]) == 5


def test_a_lookup_failure_is_reported_not_raised(monkeypatch):
    def boom(symbol):
        raise RuntimeError("network down")
    monkeypatch.setattr(sentiment.yf, "Ticker", boom)
    news = sentiment.collect_recent_news("AAA")
    assert news["articles"] == []
    assert "News lookup failed" in news["error"]


def test_an_empty_window_is_reported_rather_than_scored(patched, monkeypatch):
    """No news in two days is a real answer; a manufactured neutral is not."""
    monkeypatch.setattr(sentiment, "AZURE_INFERENCE_ENDPOINT", "https://example")
    monkeypatch.setattr(sentiment, "AZURE_INFERENCE_CREDENTIAL", "key")
    patched([_story(100), _story(200)])

    called = []
    monkeypatch.setattr(sentiment, "AzureAIChatCompletionsModel",
                        lambda **kw: called.append(kw))

    out = sentiment.get_recent_sentiment("AAA", days=2)
    assert "No AAA headlines" in out["error"]
    assert called == [], "the model must not be called with an empty window"


def test_missing_credentials_are_reported_before_any_fetch(monkeypatch):
    monkeypatch.setattr(sentiment, "AZURE_INFERENCE_ENDPOINT", "")
    monkeypatch.setattr(sentiment, "AZURE_INFERENCE_CREDENTIAL", "")
    assert "Missing Azure" in sentiment.get_recent_sentiment("AAA")["error"]


def test_prompt_text_renders_a_scored_window():
    payload = {
        "data": {"label": "positive", "score": 0.6, "confidence": 0.8,
                 "positive_factors": ["strong guidance"], "negative_factors": []},
        "articles": [{"publisher": "Wire", "title": "Beat expectations"}],
        "window": {"days": 2},
    }
    text = sentiment.sentiment_prompt_text(payload)
    assert "positive" in text and "+0.60" in text
    assert "strong guidance" in text
    assert "Beat expectations" in text


def test_prompt_text_passes_the_error_through():
    text = sentiment.sentiment_prompt_text(
        {"error": "No AAA headlines published in the last 2 days", "window": {"days": 2}})
    assert "No AAA headlines" in text

from __future__ import annotations

from gateway.realtime.openai_backend import _configured_realtime_url, _url_with_model


def test_realtime_url_defaults_to_reference_regional_endpoint(monkeypatch):
    monkeypatch.delenv("OPENAI_REALTIME_URL", raising=False)
    monkeypatch.delenv("REALTIME_API_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    assert _configured_realtime_url() == "wss://us.api.openai.com/v1/realtime"


def test_realtime_url_can_be_derived_from_openai_base_url(monkeypatch):
    monkeypatch.delenv("OPENAI_REALTIME_URL", raising=False)
    monkeypatch.delenv("REALTIME_API_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")

    assert _configured_realtime_url() == "wss://example.test/v1/realtime"


def test_realtime_url_explicit_env_takes_precedence(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://base.example.test/v1")
    monkeypatch.setenv("REALTIME_API_URL", "wss://legacy.example.test/v1/realtime")
    monkeypatch.setenv("OPENAI_REALTIME_URL", "wss://explicit.example.test/realtime")

    assert _configured_realtime_url() == "wss://explicit.example.test/realtime"


def test_realtime_model_query_param_is_added_once():
    assert (
        _url_with_model("wss://example.test/v1/realtime", "gpt-realtime-2")
        == "wss://example.test/v1/realtime?model=gpt-realtime-2"
    )
    assert (
        _url_with_model("wss://example.test/v1/realtime?model=already", "ignored")
        == "wss://example.test/v1/realtime?model=already"
    )

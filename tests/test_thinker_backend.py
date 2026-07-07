"""The OpenAI thinker must target the regional endpoint by default.

A bare AsyncOpenAI() hits api.openai.com, which 401s ("incorrect_hostname")
for accounts pinned to a regional host like us.api.openai.com — while the
gateway's realtime backend already defaults to the regional host. The thinker
must mirror that default and honor OPENAI_BASE_URL when set.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openai")  # only meaningful with the openai extra installed

from business.thinker import _DEFAULT_THINKER_BASE_URL, OpenAIThinkerModel
from business.tools.cafe_tools import build_cafe_toolset


def test_thinker_defaults_to_regional_base_url(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    model = OpenAIThinkerModel(build_cafe_toolset())
    assert str(model._client.base_url).rstrip("/") == _DEFAULT_THINKER_BASE_URL


def test_thinker_honors_openai_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://custom.example/v1")
    model = OpenAIThinkerModel(build_cafe_toolset())
    assert str(model._client.base_url).rstrip("/") == "https://custom.example/v1"


def test_empty_openai_base_url_falls_back_to_regional(monkeypatch):
    # Compose forwards OPENAI_BASE_URL=${OPENAI_BASE_URL:-}, i.e. an empty
    # string when unset. That must not defeat the regional default.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    model = OpenAIThinkerModel(build_cafe_toolset())
    assert str(model._client.base_url).rstrip("/") == _DEFAULT_THINKER_BASE_URL

from __future__ import annotations

import dataclasses

import pytest

from config import Settings
from providers.factory import make_summarizer, make_transcriber


def test_make_summarizer_unknown_provider() -> None:
    with pytest.raises(ValueError, match="provider"):
        make_summarizer(Settings(), "grok", "medium")


def test_make_summarizer_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        make_summarizer(Settings(), "ollama", "ultra")


def test_make_summarizer_base_url_from_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    import providers.llm as llm_mod

    captured: dict = {}

    def fake_init(self: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(llm_mod.LLMSummarizer, "__init__", fake_init)
    make_summarizer(Settings(), "ollama", "medium")

    assert captured["base_url"] == "http://localhost:11434/v1"


def test_make_summarizer_settings_base_url_takes_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    import providers.llm as llm_mod

    captured: dict = {}

    def fake_init(self: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(llm_mod.LLMSummarizer, "__init__", fake_init)
    settings = dataclasses.replace(Settings(), base_url="http://custom:9999/v1")
    make_summarizer(settings, "ollama", "medium")

    assert captured["base_url"] == "http://custom:9999/v1"


def test_make_summarizer_api_key_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    import providers.llm as llm_mod

    captured: dict = {}

    def fake_init(self: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(llm_mod.LLMSummarizer, "__init__", fake_init)
    settings = dataclasses.replace(Settings(), api_key="sk-test")
    make_summarizer(settings, "ollama", "medium")

    assert captured["api_key"] == "sk-test"


def test_make_summarizer_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    import providers.llm as llm_mod

    captured: dict = {}

    def fake_init(self: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(llm_mod.LLMSummarizer, "__init__", fake_init)
    make_summarizer(Settings(), "ollama", "medium", model_override="llama3:latest")

    assert captured["model"] == "llama3:latest"


def test_make_summarizer_default_model_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    import providers.llm as llm_mod

    captured: dict = {}

    def fake_init(self: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(llm_mod.LLMSummarizer, "__init__", fake_init)
    settings = dataclasses.replace(Settings(), model="my-default-model")
    make_summarizer(settings, "ollama", "medium")

    assert captured["model"] == "my-default-model"


def test_make_transcriber_passes_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    import providers.whisper as whisper_mod

    captured: dict = {}

    def fake_init(self: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(whisper_mod.WhisperTranscriber, "__init__", fake_init)
    settings = dataclasses.replace(Settings(), whisper_model="medium", whisper_device="cpu")
    make_transcriber(settings)

    assert captured["model_name"] == "medium"
    assert captured["device"] == "cpu"

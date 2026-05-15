import os
from pathlib import Path

import pytest

from config import Settings


def test_defaults() -> None:
    s = Settings.load(config_path=Path("nonexistent.yaml"))
    assert s.language == "ru"
    assert s.provider == "ollama"
    assert s.api_key is None


def test_env_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEETING_SUM_MODEL", "gpt-4o")
    monkeypatch.setenv("MEETING_SUM_LANGUAGE", "en")
    s = Settings.load(config_path=Path("nonexistent.yaml"))
    assert s.model == "gpt-4o"
    assert s.language == "en"


def test_openai_api_key_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("MEETING_SUM_API_KEY", raising=False)
    s = Settings.load(config_path=Path("nonexistent.yaml"))
    assert s.api_key == "sk-test"


def test_meeting_sum_api_key_takes_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("MEETING_SUM_API_KEY", "sk-custom")
    s = Settings.load(config_path=Path("nonexistent.yaml"))
    assert s.api_key == "sk-custom"


def test_yaml_overrides_defaults(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("language: en\nmodel: mistral\n", encoding="utf-8")
    s = Settings.load(config_path=cfg)
    assert s.language == "en"
    assert s.model == "mistral"


def test_env_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: from-yaml\n", encoding="utf-8")
    monkeypatch.setenv("MEETING_SUM_MODEL", "from-env")
    s = Settings.load(config_path=cfg)
    assert s.model == "from-env"


def test_unknown_yaml_key_ignored(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("unknown_key: value\nlanguage: en\n", encoding="utf-8")
    s = Settings.load(config_path=cfg)
    assert s.language == "en"


def test_path_fields_converted(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("audio: custom/meeting.wav\n", encoding="utf-8")
    s = Settings.load(config_path=cfg)
    assert isinstance(s.audio, Path)
    assert s.audio == Path("custom/meeting.wav")

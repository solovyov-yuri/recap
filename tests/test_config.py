import os
from pathlib import Path

import pytest

from config import ConfigError, Settings


def test_defaults() -> None:
    s = Settings.load(config_path=Path("nonexistent.yaml"))
    assert s.language == "ru"
    assert s.provider == "ollama"
    assert s.api_key is None


def test_env_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECAP_MODEL", "gpt-4o")
    monkeypatch.setenv("RECAP_LANGUAGE", "en")
    s = Settings.load(config_path=Path("nonexistent.yaml"))
    assert s.model == "gpt-4o"
    assert s.language == "en"


def test_openai_api_key_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("RECAP_API_KEY", raising=False)
    s = Settings.load(config_path=Path("nonexistent.yaml"))
    assert s.api_key == "sk-test"


def test_meeting_sum_api_key_takes_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("RECAP_API_KEY", "sk-custom")
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
    monkeypatch.setenv("RECAP_MODEL", "from-env")
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


def test_summary_mode_default() -> None:
    s = Settings.load(config_path=Path("nonexistent.yaml"))
    assert s.summary_mode == "medium"


def test_summary_mode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECAP_SUMMARY_MODE", "brief")
    s = Settings.load(config_path=Path("nonexistent.yaml"))
    assert s.summary_mode == "brief"


def test_malformed_yaml(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("key: [unclosed bracket\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        Settings.load(config_path=cfg)


def test_invalid_yaml_not_a_dict(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML mapping"):
        Settings.load(config_path=cfg)


def test_invalid_provider(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("provider: grok\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="provider"):
        Settings.load(config_path=cfg)


def test_invalid_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECAP_PROVIDER", "bad-provider")
    with pytest.raises(ConfigError, match="provider"):
        Settings.load(config_path=Path("nonexistent.yaml"))


def test_invalid_summary_mode(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("summary_mode: ultra\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="summary_mode"):
        Settings.load(config_path=cfg)


def test_invalid_max_transcript_chars_not_a_number(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("max_transcript_chars: abc\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="max_transcript_chars"):
        Settings.load(config_path=cfg)


def test_invalid_max_transcript_chars_zero(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("max_transcript_chars: 0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="max_transcript_chars"):
        Settings.load(config_path=cfg)


def test_invalid_max_transcript_chars_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECAP_MAX_TRANSCRIPT_CHARS", "-500")
    with pytest.raises(ConfigError, match="max_transcript_chars"):
        Settings.load(config_path=Path("nonexistent.yaml"))


def test_invalid_llm_timeout_not_a_number(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("llm_timeout_seconds: abc\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="llm_timeout_seconds"):
        Settings.load(config_path=cfg)


def test_invalid_llm_timeout_zero(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("llm_timeout_seconds: 0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="llm_timeout_seconds"):
        Settings.load(config_path=cfg)


def test_invalid_llm_retries_not_a_number(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("llm_retries: abc\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="llm_retries"):
        Settings.load(config_path=cfg)


def test_invalid_llm_retries_negative(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("llm_retries: -1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="llm_retries"):
        Settings.load(config_path=cfg)


def test_llm_timeout_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECAP_LLM_TIMEOUT", "120")
    s = Settings.load(config_path=Path("nonexistent.yaml"))
    assert s.llm_timeout_seconds == 120.0


def test_llm_retries_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECAP_LLM_RETRIES", "0")
    s = Settings.load(config_path=Path("nonexistent.yaml"))
    assert s.llm_retries == 0


def test_invalid_whisper_device(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("whisper_device: gpu\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="whisper_device"):
        Settings.load(config_path=cfg)


def test_invalid_whisper_compute_type(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("whisper_compute_type: bfloat16\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="whisper_compute_type"):
        Settings.load(config_path=cfg)


def test_invalid_whisper_beam_size_not_a_number(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("whisper_beam_size: fast\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="whisper_beam_size"):
        Settings.load(config_path=cfg)


def test_invalid_whisper_beam_size_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECAP_WHISPER_BEAM_SIZE", "0")
    with pytest.raises(ConfigError, match="whisper_beam_size"):
        Settings.load(config_path=Path("nonexistent.yaml"))


def test_whisper_device_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECAP_WHISPER_DEVICE", "cpu")
    s = Settings.load(config_path=Path("nonexistent.yaml"))
    assert s.whisper_device == "cpu"


def test_whisper_vad_filter_bool_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECAP_WHISPER_VAD_FILTER", "false")
    s = Settings.load(config_path=Path("nonexistent.yaml"))
    assert s.whisper_vad_filter is False


def test_invalid_chunking_mode(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("chunking_mode: stream\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="chunking_mode"):
        Settings.load(config_path=cfg)


def test_chunking_mode_truncate_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECAP_CHUNKING_MODE", "truncate")
    s = Settings.load(config_path=Path("nonexistent.yaml"))
    assert s.chunking_mode == "truncate"

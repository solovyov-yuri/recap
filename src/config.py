from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    pass


PROVIDER_PRESETS: dict[str, str | None] = {
    "openai":    None,
    "ollama":    "http://localhost:11434/v1",
    "lm-studio": "http://localhost:1234/v1",
    "vllm":      "http://localhost:8000/v1",
}

_KNOWN_FIELDS = {
    "audio", "transcript", "summary", "language",
    "whisper_model", "provider", "model", "api_key", "base_url",
    "max_transcript_chars", "summary_mode", "privacy_ack",
    "llm_timeout_seconds", "llm_retries",
}

_ENV_MAP: dict[str, str] = {
    "RECAP_AUDIO":               "audio",
    "RECAP_TRANSCRIPT":          "transcript",
    "RECAP_SUMMARY":             "summary",
    "RECAP_LANGUAGE":            "language",
    "RECAP_WHISPER_MODEL":       "whisper_model",
    "RECAP_PROVIDER":            "provider",
    "RECAP_MODEL":               "model",
    "RECAP_API_KEY":             "api_key",
    "RECAP_BASE_URL":            "base_url",
    "RECAP_MAX_TRANSCRIPT_CHARS": "max_transcript_chars",
    "RECAP_SUMMARY_MODE":         "summary_mode",
    "RECAP_PRIVACY_ACK":          "privacy_ack",
    "RECAP_LLM_TIMEOUT":          "llm_timeout_seconds",
    "RECAP_LLM_RETRIES":          "llm_retries",
}


@dataclass(frozen=True)
class Settings:
    audio: Path = Path("data/meeting.wav")
    transcript: Path = Path("data/transcript.txt")
    summary: Path = Path("data/summary.txt")
    language: str = "ru"
    whisper_model: str = "large-v3"
    provider: str = "ollama"
    model: str = "qwen3.5:latest"
    api_key: str | None = None
    base_url: str | None = None
    max_transcript_chars: int = 60_000
    summary_mode: str = "medium"
    privacy_ack: bool = False
    llm_timeout_seconds: float = 60.0
    llm_retries: int = 2

    @classmethod
    def load(cls, config_path: Path = Path("config.yaml")) -> Settings:
        data: dict = {}

        if config_path.exists():
            import yaml  # noqa: PLC0415

            with config_path.open(encoding="utf-8") as f:
                try:
                    raw = yaml.safe_load(f)
                except yaml.YAMLError as exc:
                    raise ConfigError(f"config.yaml is not valid YAML: {exc}") from exc
            if raw is not None and not isinstance(raw, dict):
                raise ConfigError(f"config.yaml must be a YAML mapping, got {type(raw).__name__}")
            data = raw or {}
            for key in set(data) - _KNOWN_FIELDS:
                logger.warning("config.yaml: unknown key %r (ignored)", key)
            data = {k: v for k, v in data.items() if k in _KNOWN_FIELDS}

        for env_var, field in _ENV_MAP.items():
            if (value := os.environ.get(env_var)) is not None:
                data[field] = value

        if "api_key" not in data and (value := os.environ.get("OPENAI_API_KEY")):
            data["api_key"] = value

        for field in ("audio", "transcript", "summary"):
            if field in data:
                try:
                    data[field] = Path(data[field])
                except TypeError:
                    raise ConfigError(f"'{field}' must be a file path string, got {data[field]!r}")

        if "max_transcript_chars" in data:
            try:
                data["max_transcript_chars"] = int(data["max_transcript_chars"])
            except (ValueError, TypeError):
                raise ConfigError(
                    f"'max_transcript_chars' must be an integer, got {data['max_transcript_chars']!r}"
                )
            if data["max_transcript_chars"] <= 0:
                raise ConfigError(
                    f"'max_transcript_chars' must be a positive integer, got {data['max_transcript_chars']}"
                )

        if "llm_timeout_seconds" in data:
            try:
                data["llm_timeout_seconds"] = float(data["llm_timeout_seconds"])
            except (ValueError, TypeError):
                raise ConfigError(
                    f"'llm_timeout_seconds' must be a number, got {data['llm_timeout_seconds']!r}"
                )
            if data["llm_timeout_seconds"] <= 0:
                raise ConfigError(
                    f"'llm_timeout_seconds' must be positive, got {data['llm_timeout_seconds']}"
                )

        if "llm_retries" in data:
            try:
                data["llm_retries"] = int(data["llm_retries"])
            except (ValueError, TypeError):
                raise ConfigError(
                    f"'llm_retries' must be a non-negative integer, got {data['llm_retries']!r}"
                )
            if data["llm_retries"] < 0:
                raise ConfigError(
                    f"'llm_retries' must be >= 0, got {data['llm_retries']}"
                )

        if "privacy_ack" in data:
            val = data["privacy_ack"]
            data["privacy_ack"] = val.lower() in ("true", "1", "yes") if isinstance(val, str) else bool(val)

        if "provider" in data and data["provider"] not in PROVIDER_PRESETS:
            available = ", ".join(PROVIDER_PRESETS)
            raise ConfigError(f"'provider' must be one of: {available}. Got {data['provider']!r}")

        if "summary_mode" in data:
            from prompts import PROMPTS  # noqa: PLC0415

            if data["summary_mode"] not in PROMPTS:
                available = ", ".join(sorted(PROMPTS))
                raise ConfigError(f"'summary_mode' must be one of: {available}. Got {data['summary_mode']!r}")

        return cls(**data)

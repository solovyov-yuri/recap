from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

PROVIDER_PRESETS: dict[str, str | None] = {
    "openai":    None,
    "ollama":    "http://localhost:11434/v1",
    "lm-studio": "http://localhost:1234/v1",
    "vllm":      "http://localhost:8000/v1",
}

_KNOWN_FIELDS = {
    "audio", "transcript", "summary", "language",
    "whisper_model", "provider", "model", "api_key", "base_url",
    "max_transcript_chars",
}

_ENV_MAP: dict[str, str] = {
    "MEETING_SUM_AUDIO":               "audio",
    "MEETING_SUM_TRANSCRIPT":          "transcript",
    "MEETING_SUM_SUMMARY":             "summary",
    "MEETING_SUM_LANGUAGE":            "language",
    "MEETING_SUM_WHISPER_MODEL":       "whisper_model",
    "MEETING_SUM_PROVIDER":            "provider",
    "MEETING_SUM_MODEL":               "model",
    "MEETING_SUM_API_KEY":             "api_key",
    "MEETING_SUM_BASE_URL":            "base_url",
    "MEETING_SUM_MAX_TRANSCRIPT_CHARS": "max_transcript_chars",
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

    @classmethod
    def load(cls, config_path: Path = Path("config.yaml")) -> Settings:
        data: dict = {}

        if config_path.exists():
            import yaml  # noqa: PLC0415

            with config_path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
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
                data[field] = Path(data[field])
        if "max_transcript_chars" in data:
            data["max_transcript_chars"] = int(data["max_transcript_chars"])

        return cls(**data)

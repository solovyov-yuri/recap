from __future__ import annotations

import logging
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

    @classmethod
    def load(cls, config_path: Path = Path("config.yaml")) -> Settings:
        if not config_path.exists():
            return cls()

        import yaml  # noqa: PLC0415

        with config_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        for key in set(data) - _KNOWN_FIELDS:
            logger.warning("config.yaml: unknown key %r (ignored)", key)

        known = {k: v for k, v in data.items() if k in _KNOWN_FIELDS}
        for field in ("audio", "transcript", "summary"):
            if field in known:
                known[field] = Path(known[field])

        return cls(**known)

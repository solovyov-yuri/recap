from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    audio: Path = Path("data/meeting.wav")
    transcript: Path = Path("data/transcript.txt")
    summary: Path = Path("data/summary.txt")
    language: str = "ru"
    whisper_model: str = "large-v3"
    ollama_model: str = "qwen3.5:latest"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            audio=Path(os.environ.get("MEETING_SUM_AUDIO", "data/meeting.wav")),
            transcript=Path(os.environ.get("MEETING_SUM_TRANSCRIPT", "data/transcript.txt")),
            summary=Path(os.environ.get("MEETING_SUM_SUMMARY", "data/summary.txt")),
            language=os.environ.get("MEETING_SUM_LANGUAGE", "ru"),
            whisper_model=os.environ.get("MEETING_SUM_WHISPER_MODEL", "large-v3"),
            ollama_model=os.environ.get("MEETING_SUM_OLLAMA_MODEL", "qwen3.5:latest"),
        )

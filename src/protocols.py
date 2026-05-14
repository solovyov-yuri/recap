from __future__ import annotations

from pathlib import Path
from typing import Protocol

from transcript import Transcript


class Transcriber(Protocol):
    def transcribe(self, audio: Path, language: str) -> Transcript: ...


class Summarizer(Protocol):
    def summarize(self, transcript_text: str) -> str: ...

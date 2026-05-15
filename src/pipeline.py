from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from protocols import Summarizer, Transcriber
from transcript import Transcript

logger = logging.getLogger(__name__)


def run_pipeline(
    audio: Path,
    transcriber: Transcriber,
    summarizer: Summarizer,
    formatter: Callable[[str], str],
    language: str = "ru",
) -> tuple[Transcript, str]:
    """Transcribe audio and produce a formatted summary. Pure — no disk I/O."""
    logger.info("Transcribing %s (language=%s)…", audio, language)
    transcript = transcriber.transcribe(audio, language)
    logger.info("Got %d segments, summarizing…", len(transcript.segments))

    raw = summarizer.summarize(transcript.to_text())
    return transcript, formatter(raw)

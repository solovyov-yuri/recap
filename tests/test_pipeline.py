from pathlib import Path

from pipeline import run_pipeline
from transcript import Segment, Transcript


class _FakeTranscriber:
    def transcribe(self, audio: Path, language: str) -> Transcript:
        return Transcript(
            segments=(
                Segment(0.0, 1.0, "Обсудили бюджет"),
                Segment(1.0, 2.0, "Решили взять Иванова"),
            )
        )


class _FakeSummarizer:
    def summarize(self, transcript_text: str) -> str:
        assert "Обсудили бюджет" in transcript_text
        assert "Решили взять Иванова" in transcript_text
        return "**Саммари**\n1. Бюджет утверждён"


def test_pipeline_returns_transcript_and_summary() -> None:
    tr, summary = run_pipeline(
        audio=Path("fake.wav"),
        transcriber=_FakeTranscriber(),
        summarizer=_FakeSummarizer(),
        formatter=lambda t: t,
        language="ru",
    )
    assert len(tr.segments) == 2
    assert "**Саммари**" in summary


def test_pipeline_applies_formatter() -> None:
    _, summary = run_pipeline(
        audio=Path("fake.wav"),
        transcriber=_FakeTranscriber(),
        summarizer=_FakeSummarizer(),
        formatter=str.upper,
    )
    assert summary.startswith("**САММАРИ**")


def test_pipeline_passes_language_to_transcriber() -> None:
    received: list[str] = []

    class _CapturingTranscriber:
        def transcribe(self, audio: Path, language: str) -> Transcript:
            received.append(language)
            return Transcript(segments=())

    class _NullSummarizer:
        def summarize(self, transcript_text: str) -> str:
            return ""

    run_pipeline(
        audio=Path("x.wav"),
        transcriber=_CapturingTranscriber(),
        summarizer=_NullSummarizer(),
        formatter=lambda t: t,
        language="en",
    )
    assert received == ["en"]

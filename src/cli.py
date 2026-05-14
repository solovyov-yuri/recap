from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

sys.stdout.reconfigure(write_through=True)

app = typer.Typer(help="Meeting transcription and summarization tool")

_DEFAULT_AUDIO = Path("meeting.wav")
_DEFAULT_TRANSCRIPT = Path("transcript.txt")
_DEFAULT_SUMMARY = Path("summary.txt")
_DEFAULT_LANGUAGE = "ru"
_DEFAULT_MODEL = "qwen3.5:latest"


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
    )


@app.command()
def transcribe(
    audio: Annotated[Path, typer.Argument(help="Audio file to transcribe")] = _DEFAULT_AUDIO,
    output: Annotated[Path, typer.Option("-o", "--output", help="Output transcript file")] = _DEFAULT_TRANSCRIPT,
    language: Annotated[str, typer.Option("-l", "--language", help="Language code (ru, en, …)")] = _DEFAULT_LANGUAGE,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Show progress logs")] = False,
) -> None:
    """Transcribe an audio file to a timestamped transcript."""
    _configure_logging(verbose)
    try:
        from providers.whisper import WhisperTranscriber  # noqa: PLC0415

        transcript = WhisperTranscriber().transcribe(audio, language)
    except Exception as exc:
        typer.echo(f"Ошибка транскрибации: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    output.write_text(transcript.to_file_format(), encoding="utf-8")
    typer.echo(f"Транскрипция сохранена в {output}")


@app.command()
def summarize(
    transcript: Annotated[Path, typer.Argument(help="Transcript file to summarize")] = _DEFAULT_TRANSCRIPT,
    output: Annotated[Path, typer.Option("-o", "--output", help="Output summary file")] = _DEFAULT_SUMMARY,
    model: Annotated[str, typer.Option("-m", "--model", help="Ollama model name")] = _DEFAULT_MODEL,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Show progress logs")] = False,
) -> None:
    """Generate a Telegram-formatted meeting summary from a transcript."""
    _configure_logging(verbose)
    try:
        from formatters import to_telegram  # noqa: PLC0415
        from providers.ollama import OllamaSummarizer  # noqa: PLC0415
        from transcript import Transcript  # noqa: PLC0415

        tr = Transcript.from_file(transcript)
        raw = OllamaSummarizer(model=model).summarize(tr.to_text())
        summary = to_telegram(raw)
    except Exception as exc:
        typer.echo(f"Ошибка генерации саммари: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(summary)
    output.write_text(summary, encoding="utf-8")
    typer.echo(f"\nСаммари сохранено в {output}")


@app.command()
def run(
    audio: Annotated[Path, typer.Argument(help="Audio file to process")] = _DEFAULT_AUDIO,
    language: Annotated[str, typer.Option("-l", "--language")] = _DEFAULT_LANGUAGE,
    model: Annotated[str, typer.Option("-m", "--model")] = _DEFAULT_MODEL,
    transcript: Annotated[Path, typer.Option("--transcript")] = _DEFAULT_TRANSCRIPT,
    summary: Annotated[Path, typer.Option("--summary")] = _DEFAULT_SUMMARY,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    """Run the full pipeline: transcribe audio, then summarize."""
    transcribe(audio=audio, output=transcript, language=language, verbose=verbose)
    summarize(transcript=transcript, output=summary, model=model, verbose=verbose)

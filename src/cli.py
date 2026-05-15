from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

sys.stdout.reconfigure(write_through=True)

app = typer.Typer(help="Meeting transcription and summarization tool")
logger = logging.getLogger(__name__)

_DEFAULT_AUDIO = Path("data/input/meeting.wav")
_DEFAULT_TRANSCRIPT = Path("data/output/transcript.txt")
_DEFAULT_SUMMARY = Path("data/output/summary.txt")
_DEFAULT_LANGUAGE = "ru"
_DEFAULT_MODEL = "qwen3.5:latest"

_AudioArg = Annotated[
    Path,
    typer.Argument(exists=True, file_okay=True, dir_okay=False, help="Audio file to process"),
]
_TranscriptArg = Annotated[
    Path,
    typer.Argument(exists=True, file_okay=True, dir_okay=False, help="Transcript file to summarize"),
]


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
    )


def _ensure_output(path: Path) -> None:
    if path.is_dir():
        typer.echo(f"Error: output path is a directory: {path}", err=True)
        raise typer.Exit(code=1)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Created output directory: %s", path.parent)


@app.command()
def transcribe(
    audio: _AudioArg = _DEFAULT_AUDIO,
    output: Annotated[Path, typer.Option("-o", "--output", help="Output transcript file")] = _DEFAULT_TRANSCRIPT,
    language: Annotated[str, typer.Option("-l", "--language", help="Language code (ru, en, …)")] = _DEFAULT_LANGUAGE,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Show progress logs")] = False,
) -> None:
    """Transcribe an audio file to a timestamped transcript."""
    _configure_logging(verbose)
    _ensure_output(output)
    logger.info("Transcribing: %s", audio)
    try:
        from providers.whisper import WhisperTranscriber  # noqa: PLC0415

        transcript = WhisperTranscriber().transcribe(audio, language)
    except Exception as exc:
        typer.echo(f"Transcription error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    logger.info("Writing transcript to %s", output.resolve())
    try:
        output.write_text(transcript.to_file_format(), encoding="utf-8")
    except OSError as exc:
        typer.echo(f"Error writing transcript to {output.resolve()}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Transcript saved to {output}")


@app.command()
def summarize(
    transcript: _TranscriptArg = _DEFAULT_TRANSCRIPT,
    output: Annotated[Path, typer.Option("-o", "--output", help="Output summary file")] = _DEFAULT_SUMMARY,
    model: Annotated[str, typer.Option("-m", "--model", help="Ollama model name")] = _DEFAULT_MODEL,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Show progress logs")] = False,
) -> None:
    """Generate a Telegram-formatted meeting summary from a transcript."""
    _configure_logging(verbose)
    _ensure_output(output)
    logger.info("Summarizing: %s", transcript)
    try:
        from formatters import to_telegram  # noqa: PLC0415
        from providers.ollama import OllamaSummarizer  # noqa: PLC0415
        from transcript import Transcript  # noqa: PLC0415

        tr = Transcript.from_file(transcript)
        raw = OllamaSummarizer(model=model).summarize(tr.to_text())
        summary = to_telegram(raw)
    except Exception as exc:
        typer.echo(f"Summarization error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(summary)
    logger.info("Writing summary to %s", output.resolve())
    try:
        output.write_text(summary, encoding="utf-8")
    except OSError as exc:
        typer.echo(f"Error writing summary to {output.resolve()}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"\nSummary saved to {output}")


@app.command()
def run(
    audio: _AudioArg = _DEFAULT_AUDIO,
    language: Annotated[str, typer.Option("-l", "--language")] = _DEFAULT_LANGUAGE,
    model: Annotated[str, typer.Option("-m", "--model")] = _DEFAULT_MODEL,
    transcript: Annotated[Path, typer.Option("--transcript")] = _DEFAULT_TRANSCRIPT,
    summary: Annotated[Path, typer.Option("--summary")] = _DEFAULT_SUMMARY,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    """Run the full pipeline: transcribe audio, then summarize."""
    logger.info("Pipeline start: %s", audio)
    transcribe(audio=audio, output=transcript, language=language, verbose=verbose)
    summarize(transcript=transcript, output=summary, model=model, verbose=verbose)
    logger.info("Pipeline done.")

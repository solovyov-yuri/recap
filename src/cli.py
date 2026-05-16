from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

sys.stdout.reconfigure(write_through=True)

app = typer.Typer(help="Meeting transcription and summarization tool")
logger = logging.getLogger(__name__)


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
    audio: Annotated[Optional[Path], typer.Argument(file_okay=True, dir_okay=False, help="Audio file to process")] = None,
    output: Annotated[Optional[Path], typer.Option("-o", "--output", help="Output transcript file")] = None,
    language: Annotated[Optional[str], typer.Option("-l", "--language", help="Language code (ru, en, …)")] = None,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Show progress logs")] = False,
) -> None:
    """Transcribe an audio file to a timestamped transcript."""
    from config import ConfigError, Settings  # noqa: PLC0415

    _configure_logging(verbose)
    try:
        settings = Settings.load()
    except ConfigError as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    audio_path = audio or settings.audio
    if not audio_path.exists():
        typer.echo(f"Error: audio file not found: {audio_path}", err=True)
        raise typer.Exit(code=1)
    output_path = output or settings.transcript
    lang = language or settings.language
    _ensure_output(output_path)
    logger.info("Transcribing: %s", audio_path)
    try:
        from providers.whisper import WhisperTranscriber  # noqa: PLC0415

        transcriber = WhisperTranscriber(settings.whisper_model)
    except Exception as exc:
        typer.echo(f"Error loading Whisper model: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        transcript = transcriber.transcribe(audio_path, lang)
    except Exception as exc:
        typer.echo(f"Transcription error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    logger.info("Writing transcript to %s", output_path.resolve())
    try:
        output_path.write_text(transcript.to_file_format(), encoding="utf-8")
    except OSError as exc:
        typer.echo(f"Error writing transcript to {output_path.resolve()}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Transcript saved to {output_path}")


@app.command()
def summarize(
    transcript: Annotated[Optional[Path], typer.Argument(file_okay=True, dir_okay=False, help="Transcript file to summarize")] = None,
    output: Annotated[Optional[Path], typer.Option("-o", "--output", help="Output summary file")] = None,
    mode: Annotated[Optional[str], typer.Option("-m", "--mode", help="Summary mode: brief | medium | detailed")] = None,
    model: Annotated[Optional[str], typer.Option("--model", help="Model name (overrides config)")] = None,
    provider: Annotated[Optional[str], typer.Option("-p", "--provider", help="Provider: openai | ollama | lm-studio | vllm")] = None,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Show progress logs")] = False,
) -> None:
    """Generate a Telegram-formatted meeting summary from a transcript."""
    from config import ConfigError, PROVIDER_PRESETS, Settings  # noqa: PLC0415
    from prompts import PROMPTS  # noqa: PLC0415

    _configure_logging(verbose)
    try:
        settings = Settings.load()
    except ConfigError as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    transcript_path = transcript or settings.transcript
    if not transcript_path.exists():
        typer.echo(f"Error: transcript file not found: {transcript_path}", err=True)
        raise typer.Exit(code=1)
    output_path = output or settings.summary
    provider_name = provider or settings.provider
    mode_name = mode or settings.summary_mode

    if provider_name not in PROVIDER_PRESETS:
        available = ", ".join(PROVIDER_PRESETS)
        typer.echo(f"Unknown provider: {provider_name!r}. Available: {available}", err=True)
        raise typer.Exit(code=1)

    if mode_name not in PROMPTS:
        available = ", ".join(PROMPTS)
        typer.echo(f"Unknown mode: {mode_name!r}. Available: {available}", err=True)
        raise typer.Exit(code=1)

    _ensure_output(output_path)
    logger.info("Summarizing: %s via %s (mode: %s)", transcript_path, provider_name, mode_name)

    from formatters import to_telegram  # noqa: PLC0415
    from providers.llm import LLMSummarizer  # noqa: PLC0415
    from transcript import Transcript  # noqa: PLC0415

    try:
        tr = Transcript.from_file(transcript_path)
    except OSError as exc:
        typer.echo(f"Error reading transcript: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        summarizer = LLMSummarizer(
            model=model or settings.model,
            api_key=settings.api_key,
            base_url=settings.base_url or PROVIDER_PRESETS[provider_name],
            max_chars=settings.max_transcript_chars,
            prompt_template=PROMPTS[mode_name],
        )
        raw = summarizer.summarize(tr.to_text())
        summary = to_telegram(raw)
    except Exception as exc:
        typer.echo(f"LLM error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(summary)
    logger.info("Writing summary to %s", output_path.resolve())
    try:
        output_path.write_text(summary, encoding="utf-8")
    except OSError as exc:
        typer.echo(f"Error writing summary to {output_path.resolve()}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"\nSummary saved to {output_path}")


@app.command()
def run(
    audio: Annotated[Optional[Path], typer.Argument(file_okay=True, dir_okay=False, help="Audio file to process")] = None,
    language: Annotated[Optional[str], typer.Option("-l", "--language")] = None,
    mode: Annotated[Optional[str], typer.Option("-m", "--mode", help="Summary mode: brief | medium | detailed")] = None,
    model: Annotated[Optional[str], typer.Option("--model", help="Model name (overrides config)")] = None,
    provider: Annotated[Optional[str], typer.Option("-p", "--provider")] = None,
    transcript: Annotated[Optional[Path], typer.Option("--transcript")] = None,
    summary: Annotated[Optional[Path], typer.Option("--summary")] = None,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    """Run the full pipeline: transcribe audio, then summarize."""
    from config import ConfigError, PROVIDER_PRESETS, Settings  # noqa: PLC0415
    from formatters import to_telegram  # noqa: PLC0415
    from prompts import PROMPTS  # noqa: PLC0415
    from providers.llm import LLMSummarizer  # noqa: PLC0415
    from providers.whisper import WhisperTranscriber  # noqa: PLC0415

    _configure_logging(verbose)
    try:
        settings = Settings.load()
    except ConfigError as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    audio_path = audio or settings.audio
    if not audio_path.exists():
        typer.echo(f"Error: audio file not found: {audio_path}", err=True)
        raise typer.Exit(code=1)

    transcript_path = transcript or settings.transcript
    summary_path = summary or settings.summary
    provider_name = provider or settings.provider
    mode_name = mode or settings.summary_mode

    if provider_name not in PROVIDER_PRESETS:
        available = ", ".join(PROVIDER_PRESETS)
        typer.echo(f"Unknown provider: {provider_name!r}. Available: {available}", err=True)
        raise typer.Exit(code=1)

    if mode_name not in PROMPTS:
        available = ", ".join(PROMPTS)
        typer.echo(f"Unknown mode: {mode_name!r}. Available: {available}", err=True)
        raise typer.Exit(code=1)

    _ensure_output(transcript_path)
    _ensure_output(summary_path)

    try:
        transcriber = WhisperTranscriber(settings.whisper_model)
    except Exception as exc:
        typer.echo(f"Error loading Whisper model: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        tr = transcriber.transcribe(audio_path, language or settings.language)
    except Exception as exc:
        typer.echo(f"Transcription error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        transcript_path.write_text(tr.to_file_format(), encoding="utf-8")
    except OSError as exc:
        typer.echo(f"Error writing transcript to {transcript_path.resolve()}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Transcript saved to {transcript_path}")

    try:
        summarizer = LLMSummarizer(
            model=model or settings.model,
            api_key=settings.api_key,
            base_url=settings.base_url or PROVIDER_PRESETS[provider_name],
            max_chars=settings.max_transcript_chars,
            prompt_template=PROMPTS[mode_name],
        )
        formatted = to_telegram(summarizer.summarize(tr.to_text()))
    except Exception as exc:
        typer.echo(f"LLM error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(formatted)
    try:
        summary_path.write_text(formatted, encoding="utf-8")
    except OSError as exc:
        typer.echo(f"Error writing summary to {summary_path.resolve()}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"\nSummary saved to {summary_path}")

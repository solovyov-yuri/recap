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


_LOCAL_HOSTNAMES = {"localhost", "127.0.0.1", "::1"}
_AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".ogg"})


def _is_external(base_url: str | None, provider: str) -> bool:
    if provider == "openai" and base_url is None:
        return True
    if base_url is None:
        return False
    from urllib.parse import urlparse  # noqa: PLC0415

    hostname = urlparse(base_url).hostname or ""
    return hostname not in _LOCAL_HOSTNAMES


def _warn_if_external(base_url: str | None, provider: str, privacy_ack: bool) -> None:
    if privacy_ack or not _is_external(base_url, provider):
        return
    endpoint = base_url or "https://api.openai.com"
    typer.echo(
        f"Warning: transcript will be sent to external endpoint ({endpoint}).\n"
        "Set 'privacy_ack: true' in config.yaml to silence this warning.",
        err=True,
    )


def _write_atomic(path: Path, text: str, label: str) -> None:
    from utils import write_text_atomic  # noqa: PLC0415

    try:
        write_text_atomic(path, text)
    except OSError as exc:
        typer.echo(f"Error writing {label} to {path.resolve()}: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _ensure_output(path: Path) -> None:
    if path.is_dir():
        typer.echo(f"Error: output path is a directory: {path}", err=True)
        raise typer.Exit(code=1)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Created output directory: %s", path.parent)


@app.command()
def batch(
    folder: Annotated[Path, typer.Argument(file_okay=False, dir_okay=True, help="Folder with audio files to process")],
    output_dir: Annotated[Optional[Path], typer.Option("-o", "--output-dir", help="Output directory (defaults to folder)")] = None,
    mode: Annotated[Optional[str], typer.Option("-m", "--mode", help="Summary mode: brief | medium | detailed")] = None,
    model: Annotated[Optional[str], typer.Option("--model", help="Model name (overrides config)")] = None,
    provider: Annotated[Optional[str], typer.Option("-p", "--provider", help="Provider: openai | ollama | lm-studio | vllm")] = None,
    language: Annotated[Optional[str], typer.Option("-l", "--language", help="Language code (ru, en, …)")] = None,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Show progress logs")] = False,
) -> None:
    """Process all audio files in a folder: transcribe and summarize each."""
    from config import ConfigError, PROVIDER_PRESETS, Settings  # noqa: PLC0415
    from formatters import to_telegram  # noqa: PLC0415
    from prompts import PROMPTS  # noqa: PLC0415
    from providers.llm import LLMSummarizer  # noqa: PLC0415
    from providers.whisper import WhisperTranscriber  # noqa: PLC0415
    from utils import write_text_atomic  # noqa: PLC0415

    _configure_logging(verbose)
    try:
        settings = Settings.load()
    except ConfigError as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not folder.is_dir():
        typer.echo(f"Error: folder not found: {folder}", err=True)
        raise typer.Exit(code=1)

    provider_name = provider or settings.provider
    mode_name = mode or settings.summary_mode
    lang = language or settings.language
    out_dir = output_dir or folder

    if provider_name not in PROVIDER_PRESETS:
        available = ", ".join(PROVIDER_PRESETS)
        typer.echo(f"Unknown provider: {provider_name!r}. Available: {available}", err=True)
        raise typer.Exit(code=1)

    if mode_name not in PROMPTS:
        available = ", ".join(PROMPTS)
        typer.echo(f"Unknown mode: {mode_name!r}. Available: {available}", err=True)
        raise typer.Exit(code=1)

    audio_files = sorted(p for p in folder.iterdir() if p.suffix.lower() in _AUDIO_EXTENSIONS)

    if not audio_files:
        typer.echo(f"No audio files found in {folder}.")
        return

    from collections import defaultdict  # noqa: PLC0415

    stem_map: defaultdict[str, list[Path]] = defaultdict(list)
    for p in audio_files:
        stem_map[p.stem].append(p)
    collisions = {stem: files for stem, files in stem_map.items() if len(files) > 1}
    if collisions:
        typer.echo("Error: output name collisions (same stem, different extension):", err=True)
        for stem, files in sorted(collisions.items()):
            typer.echo(f"  {stem!r}: {', '.join(f.name for f in files)}", err=True)
        raise typer.Exit(code=1)

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        typer.echo(f"Error creating output directory {out_dir}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _warn_if_external(settings.base_url or PROVIDER_PRESETS[provider_name], provider_name, settings.privacy_ack)

    try:
        transcriber = WhisperTranscriber(
            model_name=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            beam_size=settings.whisper_beam_size,
            vad_filter=settings.whisper_vad_filter,
            condition_on_previous_text=settings.whisper_condition_on_previous_text,
        )
    except Exception as exc:
        typer.echo(f"Error loading Whisper model: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    summarizer = LLMSummarizer(
        model=model or settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url or PROVIDER_PRESETS[provider_name],
        max_chars=settings.max_transcript_chars,
        prompt_template=PROMPTS[mode_name],
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_retries,
        chunking_mode=settings.chunking_mode,
        num_ctx=settings.num_ctx,
    )

    failures: list[tuple[Path, Exception]] = []
    succeeded = 0

    for audio_path in audio_files:
        typer.echo(f"\nProcessing: {audio_path.name}")
        transcript_path = out_dir / f"{audio_path.stem}.txt"
        summary_path = out_dir / f"{audio_path.stem}_summary.txt"
        try:
            tr = transcriber.transcribe(audio_path, lang)
            write_text_atomic(transcript_path, tr.to_file_format())
            if tr.is_empty:
                typer.echo("  No speech detected — summary skipped.", err=True)
                succeeded += 1
                continue
            raw = summarizer.summarize(tr.to_text())
            write_text_atomic(summary_path, to_telegram(raw))
            succeeded += 1
        except Exception as exc:
            typer.echo(f"  Error: {exc}", err=True)
            failures.append((audio_path, exc))

    typer.echo(f"\n{succeeded} succeeded, {len(failures)} failed.")
    if failures:
        raise typer.Exit(code=1)


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

        transcriber = WhisperTranscriber(
            model_name=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            beam_size=settings.whisper_beam_size,
            vad_filter=settings.whisper_vad_filter,
            condition_on_previous_text=settings.whisper_condition_on_previous_text,
        )
    except Exception as exc:
        typer.echo(f"Error loading Whisper model: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        transcript = transcriber.transcribe(audio_path, lang)
    except Exception as exc:
        typer.echo(f"Transcription error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    logger.info("Writing transcript to %s", output_path.resolve())
    _write_atomic(output_path, transcript.to_file_format(), "transcript")
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

    if tr.is_empty:
        typer.echo("Error: no speech detected in transcript.", err=True)
        raise typer.Exit(code=1)

    _warn_if_external(settings.base_url or PROVIDER_PRESETS[provider_name], provider_name, settings.privacy_ack)

    try:
        summarizer = LLMSummarizer(
            model=model or settings.model,
            api_key=settings.api_key,
            base_url=settings.base_url or PROVIDER_PRESETS[provider_name],
            max_chars=settings.max_transcript_chars,
            prompt_template=PROMPTS[mode_name],
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_retries,
            chunking_mode=settings.chunking_mode,
            num_ctx=settings.num_ctx,
        )
        raw = summarizer.summarize(tr.to_text())
        summary = to_telegram(raw)
    except Exception as exc:
        typer.echo(f"LLM error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(summary)
    logger.info("Writing summary to %s", output_path.resolve())
    _write_atomic(output_path, summary, "summary")
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
        transcriber = WhisperTranscriber(
            model_name=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            beam_size=settings.whisper_beam_size,
            vad_filter=settings.whisper_vad_filter,
            condition_on_previous_text=settings.whisper_condition_on_previous_text,
        )
    except Exception as exc:
        typer.echo(f"Error loading Whisper model: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        tr = transcriber.transcribe(audio_path, language or settings.language)
    except Exception as exc:
        typer.echo(f"Transcription error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _write_atomic(transcript_path, tr.to_file_format(), "transcript")
    typer.echo(f"Transcript saved to {transcript_path}")

    if tr.is_empty:
        typer.echo("No speech detected in transcript — summary skipped.", err=True)
        raise typer.Exit(code=1)

    _warn_if_external(settings.base_url or PROVIDER_PRESETS[provider_name], provider_name, settings.privacy_ack)

    try:
        summarizer = LLMSummarizer(
            model=model or settings.model,
            api_key=settings.api_key,
            base_url=settings.base_url or PROVIDER_PRESETS[provider_name],
            max_chars=settings.max_transcript_chars,
            prompt_template=PROMPTS[mode_name],
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_retries,
            chunking_mode=settings.chunking_mode,
            num_ctx=settings.num_ctx,
        )
        formatted = to_telegram(summarizer.summarize(tr.to_text()))
    except Exception as exc:
        typer.echo(f"LLM error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(formatted)
    _write_atomic(summary_path, formatted, "summary")
    typer.echo(f"\nSummary saved to {summary_path}")

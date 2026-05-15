# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`meeting-sum` is a meeting audio transcription and summarization tool. It transcribes `meeting.wav` via `faster-whisper` (CUDA) and generates a Telegram-formatted summary using a local Ollama model.

## Commands

**Install dependencies:**
```bash
uv sync --group dev   # includes pytest
```

**CLI:**
```bash
uv run meeting-sum transcribe [AUDIO]          # → transcript.txt
uv run meeting-sum summarize [TRANSCRIPT]      # → summary.txt
uv run meeting-sum run [AUDIO]                 # full pipeline
uv run meeting-sum <cmd> --help                # options per command
```

All commands accept `-v/--verbose` for progress logs, `-o/--output` for output path, `-l/--language`, `-m/--model`.

**Tests:**
```bash
uv run pytest -v
```

## Architecture

```
src/
├── cli.py           # Typer CLI — thin layer: wires providers, handles errors, does I/O
├── config.py        # @dataclass Settings + from_env() for MEETING_SUM_* env vars
├── protocols.py     # Transcriber, Summarizer (typing.Protocol)
├── pipeline.py      # run_pipeline() — pure function, no disk I/O, returns (Transcript, str)
├── transcript.py    # Segment + Transcript dataclasses; from_file / to_text / to_file_format
├── formatters.py    # to_telegram(), to_plain() — convert LLM output to Telegram Markdown
└── providers/
    ├── whisper.py   # WhisperTranscriber — lazy-imports faster_whisper after CUDA path setup
    └── ollama.py    # OllamaSummarizer + SUMMARY_PROMPT_RU constant
tests/
├── test_formatters.py
├── test_transcript.py
└── test_pipeline.py
```

`src/` is the Python path root (`tool.pytest.ini_options.pythonpath = ["src"]`, hatchling `sources = {"src" = ""}`).

## Key design rules

**Lazy imports:** `cli.py` never imports `providers.*` at top level — imports happen inside command bodies so `meeting-sum --help` is instant (no CUDA load).

**CUDA paths:** `providers/whisper.py._set_cuda_paths()` prepends `.venv` NVIDIA lib dirs to `LD_LIBRARY_PATH` before importing `faster_whisper`. Must happen before the `from faster_whisper import` line.

**Error boundary:** only `cli.py` catches exceptions — providers and pipeline let them propagate. CLI translates to `typer.Exit(code=1)` with a user-readable message.

**Pipeline is pure:** `run_pipeline()` returns `(Transcript, str)` — no file writes. CLI owns I/O.

**Adding a new LLM provider:** implement `summarize(self, transcript_text: str) -> str` — satisfies `Summarizer` protocol structurally (no inheritance needed). Wire it in `cli.py`.

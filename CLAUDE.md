# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`meeting-sum` is a meeting audio transcription and summarization tool. It transcribes audio via `faster-whisper` (CUDA) and generates a Telegram-formatted summary using any OpenAI-compatible LLM (Ollama, lm-studio, vllm, OpenAI).

## Commands

**Install dependencies:**
```bash
uv sync --group dev   # includes pytest
```

**CLI:**
```bash
uv run recap transcribe [AUDIO]          # → transcript.txt
uv run recap summarize [TRANSCRIPT]      # → summary.txt
uv run recap run [AUDIO]                 # full pipeline
uv run recap <cmd> --help                # options per command
```

All commands accept `-v/--verbose` for progress logs, `-o/--output` for output path, `-l/--language` (transcription), `--summary-language`, `-m/--mode`, `--model`, `-p/--provider`.

**Tests:**
```bash
uv run pytest -v
```

## Configuration

Settings are resolved in priority order: **CLI flags > env vars > config.yaml > defaults**.

**`config.yaml`** (optional, at project root):
```yaml
audio: data/meeting.wav
transcript: data/transcript.txt
summary: data/summary.txt
transcription_language: ru   # language passed to Whisper
summary_language: ru         # language for LLM prompts (optional, defaults to transcription_language)
whisper_model: large-v3
provider: ollama          # openai | ollama | lm-studio | vllm
model: qwen3.5:latest
api_key: null
base_url: null
max_transcript_chars: 60000
summary_mode: medium      # brief | medium | detailed
```

**Environment variables** (override config.yaml):

| Variable | Config field |
|---|---|
| `RECAP_AUDIO` | `audio` |
| `RECAP_TRANSCRIPT` | `transcript` |
| `RECAP_SUMMARY` | `summary` |
| `RECAP_TRANSCRIPTION_LANGUAGE` | `transcription_language` |
| `RECAP_SUMMARY_LANGUAGE` | `summary_language` |
| `RECAP_WHISPER_MODEL` | `whisper_model` |
| `RECAP_PROVIDER` | `provider` |
| `RECAP_MODEL` | `model` |
| `RECAP_API_KEY` | `api_key` |
| `RECAP_BASE_URL` | `base_url` |
| `RECAP_MAX_TRANSCRIPT_CHARS` | `max_transcript_chars` |
| `RECAP_SUMMARY_MODE` | `summary_mode` |
| `OPENAI_API_KEY` | `api_key` (fallback) |

`max_transcript_chars` — transcript is truncated at the last newline before this limit before being sent to the LLM (default 60 000 chars ≈ 15k tokens). A warning is logged when truncation occurs.

`summary_mode` — controls the prompt template and output structure: `brief` (2-3 sentences), `medium` (topic + discussions + decisions), `detailed` (participants + timeline + tasks with owners).

## Architecture

```
src/
├── cli.py           # Typer CLI — wires providers, calls run_pipeline(), handles I/O
├── config.py        # frozen Settings dataclass; Settings.load() reads yaml then env vars
├── protocols.py     # Transcriber, Summarizer (typing.Protocol)
├── pipeline.py      # run_pipeline() — pure function, no disk I/O, returns (Transcript, str)
├── transcript.py    # Segment + Transcript dataclasses; from_file / to_text / to_file_format
├── formatters.py    # to_telegram(), to_plain() — convert LLM output to Telegram Markdown
└── providers/
    ├── whisper.py   # WhisperTranscriber — lazy-imports faster_whisper after CUDA path setup
    └── llm.py       # LLMSummarizer — OpenAI-compatible client, streaming, transcript truncation
tests/
├── test_config.py
├── test_formatters.py
├── test_transcript.py
└── test_pipeline.py
```

`src/` is the Python path root (`tool.pytest.ini_options.pythonpath = ["src"]`, hatchling `sources = {"src" = ""}`).

## Key design rules

**Lazy imports:** `cli.py` never imports `providers.*` at top level — imports happen inside command bodies so `recap --help` is instant (no CUDA load).

**CUDA paths:** `providers/whisper.py._set_cuda_paths()` prepends `.venv` NVIDIA lib dirs to `PATH`/`LD_LIBRARY_PATH` before importing `faster_whisper`. Must happen before the `from faster_whisper import` line.

**Error boundary:** only `cli.py` catches exceptions — providers and pipeline let them propagate. CLI translates to `typer.Exit(code=1)` with a user-readable message.

**Pipeline is pure:** `run_pipeline()` returns `(Transcript, str)` — no file writes. CLI owns I/O. Both `run` and the individual commands go through `run_pipeline()`.

**LLM streaming:** `LLMSummarizer.summarize()` uses `stream=True` — tokens are printed to stderr as they arrive; the full string is returned for post-processing.

**Transcript is immutable:** `Transcript.segments` is `tuple[Segment, ...]` — `frozen=True` on the dataclass is meaningful.

**Adding a new LLM provider:** implement `summarize(self, transcript_text: str) -> str` — satisfies `Summarizer` protocol structurally (no inheritance needed). Add a preset URL to `PROVIDER_PRESETS` in `config.py`, wire in `cli.py`.

**Adding a new summary mode (same language):** add a `SUMMARY_PROMPT_<NAME>_<LANG>` constant in `prompts.py` and add it to `PROMPTS[lang][name]`. Update `SUMMARY_MODES` frozenset. No other files need changing.

**Adding a new summary language:** add system + mode constants in `prompts.py`, register them in `PROMPTS["<lang>"]` and `CHUNK_PROMPTS["<lang>"]`. Config validation picks up the new language automatically. Currently only `ru` is shipped; setting `summary_language: en` is rejected until English prompts land.

**Breaking change (v0.1):** `-m` flag was reassigned from `--model` to `--mode`. Use `--model` (long form) to specify the LLM model.

**Breaking change (v0.2):** `config.yaml` field `language` renamed to `transcription_language`. Env var `RECAP_LANGUAGE` renamed to `RECAP_TRANSCRIPTION_LANGUAGE`.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`meeting-sum` is a meeting audio transcription and summarization tool. It transcribes audio via `faster-whisper` (CUDA) and generates a Telegram-formatted summary using any OpenAI-compatible LLM (Ollama, lm-studio, vllm, OpenAI, xAI/Grok).

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

All commands accept `-v/--verbose` for progress logs, `-o/--output` for output path, `-l/--language`, `-m/--model`, `-p/--provider`.

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
language: ru
whisper_model: large-v3
provider: ollama          # openai | xai | ollama | lm-studio | vllm
model: qwen3.5:latest
api_key: null
base_url: null
max_transcript_chars: 60000
summary_mode: medium      # brief | medium | detailed
chunking_mode: chunk      # chunk | truncate
privacy_ack: false
llm_timeout_seconds: 60
llm_retries: 2
num_ctx: null             # Ollama context window size in tokens (e.g. 32768); null = model default
whisper_device: cuda      # cuda | cpu | auto
whisper_compute_type: default
whisper_beam_size: 5
whisper_vad_filter: true
whisper_condition_on_previous_text: true
```

**Environment variables** (override config.yaml):

| Variable | Config field |
|---|---|
| `RECAP_AUDIO` | `audio` |
| `RECAP_TRANSCRIPT` | `transcript` |
| `RECAP_SUMMARY` | `summary` |
| `RECAP_LANGUAGE` | `language` |
| `RECAP_WHISPER_MODEL` | `whisper_model` |
| `RECAP_PROVIDER` | `provider` |
| `RECAP_MODEL` | `model` |
| `RECAP_API_KEY` | `api_key` |
| `RECAP_BASE_URL` | `base_url` |
| `RECAP_MAX_TRANSCRIPT_CHARS` | `max_transcript_chars` |
| `RECAP_SUMMARY_MODE` | `summary_mode` |
| `RECAP_CHUNKING_MODE` | `chunking_mode` |
| `RECAP_PRIVACY_ACK` | `privacy_ack` |
| `RECAP_LLM_TIMEOUT` | `llm_timeout_seconds` |
| `RECAP_LLM_RETRIES` | `llm_retries` |
| `RECAP_NUM_CTX` | `num_ctx` |
| `RECAP_WHISPER_DEVICE` | `whisper_device` |
| `RECAP_WHISPER_COMPUTE_TYPE` | `whisper_compute_type` |
| `RECAP_WHISPER_BEAM_SIZE` | `whisper_beam_size` |
| `RECAP_WHISPER_VAD_FILTER` | `whisper_vad_filter` |
| `RECAP_WHISPER_CONDITION_ON_PREVIOUS_TEXT` | `whisper_condition_on_previous_text` |
| `OPENAI_API_KEY` | `api_key` (fallback) |

`max_transcript_chars` — transcript is truncated at the last newline before this limit before being sent to the LLM (default 60 000 chars ≈ 15k tokens). A warning is logged when truncation occurs.

`summary_mode` — controls the prompt template and output structure: `brief` (2-3 sentences), `medium` (topic + discussions + decisions), `detailed` (participants + timeline + tasks with owners).

`chunking_mode` — what to do when transcript exceeds `max_transcript_chars`: `chunk` splits into chunks summarized separately then merged (default), `truncate` cuts at the limit.

`num_ctx` — Ollama context window size in tokens passed via `options.num_ctx` in the request body. Ignored by OpenAI/xAI. Use when the default Ollama context (2 048–4 096 tokens) is too small for your transcript chunks; e.g. `32768` for ~20 000-char chunks.

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

**Adding a new summary mode:** add a `SUMMARY_PROMPT_<NAME>_RU` constant and register it in `PROMPTS` dict in `providers/llm.py`. No other files need changing.

**Breaking change (v0.1):** `-m` flag was reassigned from `--model` to `--mode`. Use `--model` (long form) to specify the LLM model.

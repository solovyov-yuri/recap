# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project overview

`meeting-sum` is a meeting audio transcription and summarization tool. It transcribes audio via `faster-whisper` (CUDA) and generates a Telegram-formatted summary using any OpenAI-compatible LLM (Ollama, lm-studio, vllm, OpenAI).

## Commands

**Install dependencies:**
```bash
uv sync --group dev   # includes pytest, ruff, mypy
```

**CLI:**
```bash
uv run recap transcribe [AUDIO]          # → transcript.txt
uv run recap summarize [TRANSCRIPT]      # → summary.txt
uv run recap run [AUDIO]                 # full pipeline
uv run recap batch [FOLDER]              # process all audio files in a folder
uv run recap <cmd> --help                # options per command
```

All commands accept `-v/--verbose` for progress logs, `-o/--output` for output path,
`-l/--language` (transcription language), `--summary-language`, `-m/--mode`,
`--model`, `-p/--provider`, `-f/--format`.

**Tests:**
```bash
uv run pytest -v                         # all tests
uv run pytest tests/integration -v      # integration tests only
uv run pytest -m integration -v         # same via marker
```

**Linting and type checking:**
```bash
uv run ruff check src/                   # lint
uv run ruff format src/                  # format
uv run mypy src/                         # type check
```

## Configuration

Settings are resolved in priority order: **CLI flags > env vars > config.yaml > defaults**.

**`config.yaml`** (optional, at project root):
```yaml
audio: data/meeting.wav
transcript: data/transcript.txt
summary: data/summary.txt
transcription_language: ru   # language passed to Whisper
summary_language: ru         # language for LLM prompts (optional, defaults to "ru")
whisper_model: large-v3
provider: ollama          # openai | ollama | lm-studio | vllm
model: qwen3.5:latest
api_key: null
base_url: null
max_transcript_chars: 60000
summary_mode: medium      # brief | medium | detailed
privacy_ack: false        # set true to suppress external-endpoint warning
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
| `RECAP_PRIVACY_ACK` | `privacy_ack` |
| `OPENAI_API_KEY` | `api_key` (fallback) |

**Breaking changes:**
- `language` field renamed to `transcription_language` (old name triggers a deprecation warning, still works).
- `RECAP_LANGUAGE` renamed to `RECAP_TRANSCRIPTION_LANGUAGE` (same deprecation behaviour).
- `-m` flag was reassigned from `--model` to `--mode`. Use `--model` (long form) for the LLM model name.

## Architecture

```
src/
├── cli.py           # Typer CLI — wires providers via factory, handles I/O, error boundary
├── config.py        # frozen Settings dataclass; Settings.load() reads yaml then env vars
├── protocols.py     # Transcriber, Summarizer (typing.Protocol)
├── pipeline.py      # run_pipeline() — pure function, no disk I/O, returns (Transcript, str)
├── transcript.py    # Segment + Transcript dataclasses; from_file / to_text / to_file_format
├── formatters.py    # to_telegram(), to_plain(), to_json() — format LLM output
├── models.py        # MeetingSummary dataclass (used by JSON formatter)
├── utils.py         # write_text_atomic() — safe file writes via tmp+rename
├── prompts.py       # PROMPTS[lang][mode] nested dict; get_prompt(); CHUNK_PROMPTS
└── providers/
    ├── factory.py   # make_summarizer(), make_transcriber() — single wiring point
    ├── whisper.py   # WhisperTranscriber — lazy-imports faster_whisper after CUDA path setup
    └── llm.py       # LLMSummarizer — OpenAI-compatible client, streaming, chunked summarization
tests/
├── test_cli.py
├── test_config.py
├── test_factory.py
├── test_formatters.py
├── test_llm.py
├── test_pipeline.py
├── test_transcript.py
├── test_utils.py
├── test_whisper.py
└── integration/
    └── test_cli_flows.py   # end-to-end CLI flows; mocks only at factory boundary
```

`src/` is the Python path root (`tool.pytest.ini_options.pythonpath = ["src"]`, hatchling `sources = {"src" = ""}`).

## Key design rules

**Lazy imports:** `cli.py` never imports `providers.*` at top level — imports happen inside command bodies so `recap --help` is instant (no CUDA load).

**Provider factory:** `providers/factory.py` is the single place that validates provider name, resolves base URL, selects prompt templates by language/mode, and constructs `LLMSummarizer` / `WhisperTranscriber`. CLI calls `make_summarizer()` and `make_transcriber()` — it does not construct providers directly.

**CUDA paths:** `providers/whisper.py._set_cuda_paths()` prepends `.venv` NVIDIA lib dirs to `PATH`/`LD_LIBRARY_PATH` before importing `faster_whisper`. Must happen before the `from faster_whisper import` line.

**Error boundary:** only `cli.py` catches exceptions — providers and pipeline let them propagate. CLI translates to `typer.Exit(code=1)` with a user-readable message.

**Pipeline is pure:** `run_pipeline()` remains a pure helper for tests/embedders: it returns `(Transcript, str)` and performs no disk I/O. CLI commands own I/O and call providers through `providers.factory`.

**Atomic writes:** all file output goes through `utils.write_text_atomic()` — writes to a temp file in the same directory, then renames atomically. Never write directly with `Path.write_text()` in CLI code.

**LLM streaming:** `LLMSummarizer.summarize()` uses `stream=True` — tokens are printed to stderr as they arrive; the full string is returned for post-processing.

**Transcript is immutable:** `Transcript.segments` is `tuple[Segment, ...]` — `frozen=True` on the dataclass is meaningful.

**Prompt structure:** `PROMPTS` is a two-level dict `{lang: {mode: (system, user_template)}}`. Currently only `"ru"` is shipped. `get_prompt(language, mode)` raises `KeyError` with a clear message for unknown language/mode. `CHUNK_PROMPTS` follows the same structure for chunked summarization.

**Adding a new LLM provider:** add a preset URL to `PROVIDER_PRESETS` in `config.py` — no other files need changing.

**Adding a new summary mode:** add prompt constants in `prompts.py` and register them in `PROMPTS["ru"][name]`. Update `SUMMARY_MODES`. No other files need changing.

**Adding a new summary language:** add system + mode constants in `prompts.py`, register in `PROMPTS["<lang>"]` and `CHUNK_PROMPTS["<lang>"]`. Config validation picks up the new language automatically.

# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project overview

`meeting-sum` is a meeting audio transcription and summarization tool. It transcribes audio via `faster-whisper` (CUDA) and generates a Telegram-formatted summary using any OpenAI-compatible LLM (OpenAI, xAI, Ollama, lm-studio, vllm).

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

Config is a **nested** schema with two sections — `transcription` and `summarization` — each holding a `model` sub-section. There is no backwards compatibility with the old flat keys: any unknown key (legacy or otherwise) raises `ConfigError` with a generic "Unknown config key" message.

**`config.yaml`** (optional, at project root):
```yaml
audio: data/meeting.wav
transcript: data/transcript.txt
summary: data/summary.txt
transcription:
  language: ru                 # language passed to Whisper
  model:
    provider: faster-whisper   # only faster-whisper is supported
    name: large-v3
    device: cuda               # cuda | cpu | auto
    compute_type: default      # default | float16 | int8 | int8_float16 | float32
    beam_size: 5
    vad_filter: true
    condition_on_previous_text: true
summarization:
  language: null               # LLM prompt language; null → "ru" (transcription.language NOT inherited)
  mode: medium                 # brief | medium | detailed
  max_transcript_chars: 60000
  timeout_seconds: 60
  retries: 2
  chunking_mode: chunk         # chunk | truncate
  model:
    provider: ollama           # openai | xai | ollama | lm-studio | vllm
    name: qwen3.5:latest
    api_key: null
    base_url: null
privacy_ack: false             # set true to suppress external-endpoint warning
```

**Environment variables** (override config.yaml; names mirror the nested path):

| Variable | Config field |
|---|---|
| `RECAP_AUDIO` | `audio` |
| `RECAP_TRANSCRIPT` | `transcript` |
| `RECAP_SUMMARY` | `summary` |
| `RECAP_PRIVACY_ACK` | `privacy_ack` |
| `RECAP_TRANSCRIPTION_LANGUAGE` | `transcription.language` |
| `RECAP_TRANSCRIPTION_MODEL_PROVIDER` | `transcription.model.provider` |
| `RECAP_TRANSCRIPTION_MODEL_NAME` | `transcription.model.name` |
| `RECAP_TRANSCRIPTION_MODEL_DEVICE` | `transcription.model.device` |
| `RECAP_TRANSCRIPTION_MODEL_COMPUTE_TYPE` | `transcription.model.compute_type` |
| `RECAP_TRANSCRIPTION_MODEL_BEAM_SIZE` | `transcription.model.beam_size` |
| `RECAP_TRANSCRIPTION_MODEL_VAD_FILTER` | `transcription.model.vad_filter` |
| `RECAP_TRANSCRIPTION_MODEL_CONDITION_ON_PREVIOUS_TEXT` | `transcription.model.condition_on_previous_text` |
| `RECAP_SUMMARIZATION_LANGUAGE` | `summarization.language` |
| `RECAP_SUMMARIZATION_MODE` | `summarization.mode` |
| `RECAP_SUMMARIZATION_MAX_TRANSCRIPT_CHARS` | `summarization.max_transcript_chars` |
| `RECAP_SUMMARIZATION_TIMEOUT_SECONDS` | `summarization.timeout_seconds` |
| `RECAP_SUMMARIZATION_RETRIES` | `summarization.retries` |
| `RECAP_SUMMARIZATION_CHUNKING_MODE` | `summarization.chunking_mode` |
| `RECAP_SUMMARIZATION_MODEL_PROVIDER` | `summarization.model.provider` |
| `RECAP_SUMMARIZATION_MODEL_NAME` | `summarization.model.name` |
| `RECAP_SUMMARIZATION_MODEL_API_KEY` | `summarization.model.api_key` |
| `RECAP_SUMMARIZATION_MODEL_BASE_URL` | `summarization.model.base_url` |

`summarization.language` defaults to `null`; the factory falls back to `"ru"`. It deliberately does not inherit `transcription.language`, so English audio still produces a Russian summary unless `summarization.language` or `--summary-language` is set explicitly.

`summarization.max_transcript_chars` is the per-request LLM limit. Long transcripts are split into chunks by default (`summarization.chunking_mode: chunk`) or truncated at the last newline before this limit when `chunking_mode: truncate`.

`summarization.mode` — controls the prompt template and output structure: `brief` (2-3 sentences), `medium` (topic + discussions + decisions), `detailed` (participants + timeline + tasks with owners).

**Breaking changes:**
- Config schema changed to nested `transcription` / `summarization` sections (each with a `model` sub-section). Old flat keys (`provider`, `model`, `whisper_model`, `summary_mode`, `max_transcript_chars`, `transcription_language`, `summary_language`, `chunking_mode`, …) are no longer supported and raise `ConfigError` with a generic "Unknown config key" message.
- Old flat `RECAP_*` env vars (`RECAP_PROVIDER`, `RECAP_MODEL`, `RECAP_WHISPER_MODEL`, `RECAP_SUMMARY_MODE`, `RECAP_MAX_TRANSCRIPT_CHARS`, `RECAP_API_KEY`, `RECAP_BASE_URL`, `RECAP_LANGUAGE`, …) are replaced by the nested-path names above. There is no `OPENAI_API_KEY` fallback — set the key via `summarization.model.api_key` or `RECAP_SUMMARIZATION_MODEL_API_KEY`.
- `-m` flag was reassigned from `--model` to `--mode`. Use `--model` (long form) for the LLM model name.

## Architecture

```
src/
├── cli.py           # Typer CLI — wires providers via factory, handles I/O, error boundary
├── config.py        # nested frozen Settings dataclasses; Settings.load() reads nested yaml then env vars
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

**Nested config:** `Settings` is composed of frozen sub-dataclasses — `TranscriptionSettings` (with `TranscriptionModelSettings`) and `SummarizationSettings` (with `SummarizationModelSettings`). `Settings.load()` rejects unknown and legacy flat keys with `ConfigError`; there is no silent remapping. All sub-dataclasses are `frozen=True`.

**Lazy imports:** `cli.py` never imports `providers.*` at top level — imports happen inside command bodies so `recap --help` is instant (no CUDA load).

**Provider factory:** `providers/factory.py` is the single place that validates provider name (LLM and transcription), resolves base URL, selects prompt templates by language/mode, and constructs `LLMSummarizer` / `WhisperTranscriber` from the nested `settings.transcription` / `settings.summarization` sections. CLI calls `make_summarizer()` and `make_transcriber()` — it does not construct providers directly.

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

**Breaking change (v0.3):** `config.yaml` switched to a nested schema (`transcription` / `summarization`, each with a `model` sub-section). Old flat keys (`provider`, `model`, `whisper_model`, `summary_mode`, `max_transcript_chars`, `transcription_language`, `summary_language`, `chunking_mode`, …) and old flat `RECAP_*` env vars are no longer supported; they raise `ConfigError`. `OPENAI_API_KEY` is no longer used as a fallback; set `summarization.model.api_key` or `RECAP_SUMMARIZATION_MODEL_API_KEY`.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`meeting-sum` is a meeting audio transcription and summarization tool. It transcribes audio via `faster-whisper` (CUDA) and generates a Telegram-formatted summary using any OpenAI-compatible LLM (Ollama, lm-studio, vllm, OpenAI, xAI/Grok).

## Commands

> **⚠️ Environment — read before running anything.**
> Developed on **Windows**; the virtualenv is a Windows venv at **`.venv\Scripts\python.exe`**
> (from WSL: `.venv/Scripts/python.exe`). **Do NOT run `uv` (`uv run`, `uv sync`, …) — it breaks the venv.**
> Do not run tests/linters/type-checks yourself; suggest the exact command and ask the user to run it.
> For a read-only smoke check, use the existing interpreter directly (`.venv/Scripts/python.exe -c "..."`), never `uv`.
> The `uv ...` lines below are commands **to hand to the user**.

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

Config is **nested** by section: top-level paths, `transcription`, `summarization`, `preprocessing`.

**`config.yaml`** (optional, at project root):
```yaml
audio: data/meeting.wav
transcript: data/transcript.txt
summary: data/summary.txt
transcription:
  language: ru                 # transcription language code
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
    num_ctx: null              # Ollama context window in tokens (e.g. 32768); null = model default
privacy_ack: false
preprocessing:
  enabled: false               # set true to preprocess audio with ffmpeg before Whisper
  sample_rate: 16000
  channels: 1                  # 1 (mono) or 2 (stereo)
  codec: pcm_s16le             # only pcm_s16le is supported
  loudness_normalization: false
  target_lufs: -16.0
  true_peak_db: -1.5
  loudness_range: 11.0
  highpass_hz: null            # positive integer or null
  keep_temp: false
```

**Environment variables** (override config.yaml; one per leaf, named after the nested path):

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
| `RECAP_SUMMARIZATION_MODEL_NUM_CTX` | `summarization.model.num_ctx` |
| `RECAP_PREPROCESSING_*` | `preprocessing.*` (one per field) |

`max_transcript_chars` — transcript is truncated at the last newline before this limit before being sent to the LLM (default 60 000 chars ≈ 15k tokens). A warning is logged when truncation occurs.

`summarization.mode` — controls the prompt template and output structure: `brief` (2-3 sentences), `medium` (topic + discussions + decisions), `detailed` (participants + timeline + tasks with owners).

`chunking_mode` — what to do when transcript exceeds `max_transcript_chars`: `chunk` splits into chunks summarized separately then merged (default), `truncate` cuts at the limit.

`num_ctx` (`summarization.model.num_ctx`) — Ollama context window size in tokens passed via `options.num_ctx` in the request body. Ignored by OpenAI/xAI. Use when the default Ollama context (2 048–4 096 tokens) is too small for your transcript chunks; e.g. `32768` for ~20 000-char chunks.

## Architecture

```
src/
├── cli.py           # Typer CLI — builds providers via factory, orchestrates I/O per command
├── config.py        # nested frozen Settings dataclasses; Settings.load() reads yaml then env vars
├── transcript.py    # Segment + Transcript dataclasses; from_file / to_text / to_file_format
├── formatters.py    # to_telegram(), to_plain(), to_json() — convert LLM output for output formats
├── prompts.py       # SYSTEM/SUMMARY/CHUNK prompt constants; PROMPTS[lang][mode], get_prompt()
├── preprocessing.py # prepared_audio() ctx manager + preprocess_audio() — ffmpeg normalization
├── models.py        # MeetingSummary dataclass (json output)
├── utils.py         # write_text_atomic()
└── providers/
    ├── factory.py   # make_transcriber() / make_summarizer() — validate + build providers
    ├── whisper.py   # WhisperTranscriber — lazy-imports faster_whisper after CUDA path setup
    └── llm.py       # LLMSummarizer — OpenAI-compatible client, streaming, chunking/truncation
tests/               # test_config, test_cli, test_factory, test_formatters, test_llm,
                     # test_transcript, test_pipeline, test_preprocessing, test_utils, test_whisper
                     # + integration/test_cli_flows
```

`src/` is the Python path root (`tool.pytest.ini_options.pythonpath = ["src"]`, hatchling `sources = {"src" = ""}`).

## Key design rules

**Lazy imports:** `cli.py` never imports `providers.*` at top level — imports happen inside command bodies so `recap --help` is instant (no CUDA load).

**CUDA paths:** `providers/whisper.py._set_cuda_paths()` prepends `.venv` NVIDIA lib dirs to `PATH`/`LD_LIBRARY_PATH` before importing `faster_whisper`. Must happen before the `from faster_whisper import` line.

**Error boundary:** only `cli.py` catches exceptions — providers and pipeline let them propagate. CLI translates to `typer.Exit(code=1)` with a user-readable message.

**Orchestration lives in the CLI:** each command (`run`/`summarize`/`batch`) drives transcribe → write transcript → summarize → format → write summary itself, so it can write the transcript to disk before the LLM call (surviving an LLM failure), short-circuit on empty transcription, and branch on output format. Providers are built through `providers.factory`; the CLI is the only error boundary.

**LLM streaming:** `LLMSummarizer.summarize()` uses `stream=True` — tokens are printed to stderr as they arrive; the full string is returned for post-processing.

**Transcript is immutable:** `Transcript.segments` is `tuple[Segment, ...]` — `frozen=True` on the dataclass is meaningful.

**Adding a new LLM provider:** implement `summarize(self, transcript_text: str) -> str` — satisfies `Summarizer` protocol structurally (no inheritance needed). Add a preset URL to `PROVIDER_PRESETS` in `config.py`; it is then accepted by `make_summarizer()` in `providers/factory.py`.

**Adding a new summary mode:** add a `SUMMARY_PROMPT_<NAME>_RU` constant in `prompts.py`, register it under `PROMPTS[lang][mode]`, and add the mode name to `SUMMARY_MODES`. No other files need changing.

**Breaking change (v0.1):** `-m` flag was reassigned from `--model` to `--mode`. Use `--model` (long form) to specify the LLM model.

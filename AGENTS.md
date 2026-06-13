# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

## Project overview

`recap` transcribes meeting audio via `faster-whisper` (CUDA) and generates a Telegram-formatted summary
using any OpenAI-compatible LLM (OpenAI, xAI, Ollama, lm-studio, vllm).

## Environment & commands

The project is cross-platform and is normally driven with `uv` (e.g. `uv run pytest -v`).

The development machine here is **Windows**, with the venv under `.venv\Scripts\` (reached from WSL as
`.venv/Scripts/`). From WSL, do **not** run `uv` — it rebuilds and corrupts the Windows venv. Run the venv's
executables directly; you may run tests/lint/types yourself this way:

```bash
.venv/Scripts/pytest.exe -v           # tests (append tests/integration for integration only)
.venv/Scripts/ruff.exe check src/     # lint  (ruff.exe format src/ to format)
.venv/Scripts/mypy.exe src/           # type check
.venv/Scripts/recap.exe <cmd> --help  # CLI; commands: transcribe, summarize, run, batch, preprocess
```

**Git:** never commit, create branches, or make other git changes without an explicit request from the user.

## Architecture

```
src/
├── cli.py           # Typer CLI: per-command orchestration, I/O, the only error boundary
├── config.py        # nested frozen Settings dataclasses + Settings.load() (yaml → env, strict validation)
├── transcript.py    # Segment + Transcript (frozen); from_file / to_text / to_file_format
├── formatters.py    # to_telegram(), to_plain(), to_json()
├── models.py        # MeetingSummary dataclass (JSON output)
├── utils.py         # write_text_atomic()
├── preprocessing.py # preprocess_audio() + prepared_audio() context manager (ffmpeg)
├── prompts.py       # PROMPTS[lang][mode] + CHUNK_PROMPTS; get_prompt(); SUMMARY_MODES
└── providers/
    ├── factory.py   # make_summarizer() / make_transcriber() — the single provider wiring point
    ├── whisper.py   # WhisperTranscriber (lazy faster_whisper import after CUDA path setup)
    └── llm.py       # LLMSummarizer (OpenAI-compatible client, streaming, chunking/truncation)
tests/               # one test_*.py per module + integration/test_cli_flows.py (mocks only at factory boundary)
```

`src/` is the Python path root (`pythonpath = ["src"]`, hatchling `sources = {"src" = ""}`).

## Configuration

Resolution order: **CLI flags > env vars > config.yaml > defaults**.

The schema is **nested**: top-level paths + `privacy_ack`, plus `transcription`, `summarization`, and
`preprocessing` sections. `transcription` and `summarization` each hold a `model` sub-section. Validation is
strict — any unknown key (including old flat keys) raises `ConfigError`.

- **Full template:** `config.yaml.example`. Don't reproduce it here.
- **Env vars:** `RECAP_` + the upper-snake-cased nested path (e.g. `summarization.model.num_ctx` →
  `RECAP_SUMMARIZATION_MODEL_NUM_CTX`). The authoritative list is the `_ENV_*` maps in `config.py`.
  There is no `OPENAI_API_KEY` fallback — set `summarization.model.api_key` / `RECAP_SUMMARIZATION_MODEL_API_KEY`.

Non-obvious semantics:
- `summarization.language` defaults to `null` → the factory uses `"ru"`. It does **not** inherit
  `transcription.language`, so English audio still yields a Russian summary unless set explicitly.
- `summarization.max_transcript_chars` is the per-request LLM limit. `chunking_mode: chunk` splits long
  transcripts and merges per-chunk summaries; `truncate` cuts at the last newline before the limit.
- `summarization.model.num_ctx` is Ollama's `options.num_ctx`; ignored by OpenAI/xAI.
- `summarization.mode`: `brief` (2–3 sentences) | `medium` (topic + discussions + decisions) |
  `detailed` (participants + timeline + tasks with owners).

## Key design rules

- **Strict nested config.** `Settings` and all sub-sections are `frozen=True`. `Settings.load()` rejects unknown
  keys; there is no silent remapping of legacy keys.
- **Factory is the only wiring point.** CLI calls `make_summarizer()` / `make_transcriber()` and never constructs
  providers directly. The factory validates provider/mode/language, resolves base URL, and picks prompts.
- **CLI owns everything around the providers.** Each command drives transcribe → write transcript → summarize →
  format → write summary, so it persists the transcript before the LLM call (surviving an LLM failure),
  short-circuits on empty transcription, and branches on output format. `cli.py` is the *only* place that catches
  exceptions, translating them to `typer.Exit(code=1)`; providers let exceptions propagate.
- **Lazy imports.** `cli.py` imports `providers.*` inside command bodies, so `recap --help` stays instant (no CUDA load).
- **CUDA paths.** `whisper._set_cuda_paths()` must prepend the `.venv` NVIDIA lib dirs before the `from faster_whisper import` line.
- **Atomic writes.** All file output goes through `utils.write_text_atomic()` — never `Path.write_text()` in CLI code.
- **LLM streaming.** `LLMSummarizer.summarize()` streams tokens to stderr and returns the full string.
- **Immutable transcript.** `Transcript.segments` is `tuple[Segment, ...]`.
- **Preprocessing.** The `preprocess` command always runs ffmpeg (ignores `enabled`, since invoking it is itself
  the request); `transcribe`/`run`/`batch` use `prepared_audio()`, which respects `enabled`.

Extending:
- **New OpenAI-compatible provider:** add its preset URL to `PROVIDER_PRESETS` in `config.py` — nothing else changes.
- **New summary mode:** add prompt constants in `prompts.py`, register under `PROMPTS["ru"][name]`, add the name to `SUMMARY_MODES`.
- **New summary language:** add constants in `prompts.py`, register under `PROMPTS["<lang>"]` and `CHUNK_PROMPTS["<lang>"]`; config validation picks it up automatically.

"""Anti-regression guards.

These exist because a bad merge once left `config.py` referencing undefined names,
so *every* `import config` raised NameError and the whole CLI was dead on arrival —
yet nothing failed loudly until a command was run. A cheap import smoke test plus a
source-level guard on cli.py catch that class of breakage immediately.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

# Every importable module under src/. Importing must not execute heavy work
# (faster_whisper / CUDA / network) — those are deferred into method bodies.
_MODULES = [
    "config",
    "cli",
    "transcript",
    "formatters",
    "prompts",
    "preprocessing",
    "models",
    "utils",
    "providers.factory",
    "providers.llm",
    "providers.whisper",
]


@pytest.mark.parametrize("module_name", _MODULES)
def test_module_imports(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


def test_settings_load_with_defaults() -> None:
    from config import Settings

    # Must not raise even with no config file present.
    Settings.load(config_path=Path("nonexistent.yaml"))


def test_cli_has_no_flat_settings_references() -> None:
    """cli.py must build providers only via the factory, never the removed flat fields.

    The flat attributes below (settings.model / .api_key / ...) belonged to a pre-nested
    Settings and no longer exist; referencing them is the exact bug that broke `main`.
    """
    src = (Path(__file__).resolve().parents[1] / "src" / "cli.py").read_text(encoding="utf-8")
    forbidden = [
        "settings.model",
        "settings.api_key",
        "settings.base_url",
        "settings.max_transcript_chars",
        "settings.llm_timeout_seconds",
        "settings.llm_retries",
        "settings.chunking_mode",
        "settings.num_ctx",
        "LLMSummarizer(",  # cli must go through make_summarizer, not construct directly
    ]
    hits = [token for token in forbidden if token in src]
    assert not hits, f"cli.py references removed flat config / bypasses factory: {hits}"

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from cli import app

runner = CliRunner()


def test_help_no_cuda_import() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "transcribe" in result.output


def test_transcribe_help() -> None:
    result = runner.invoke(app, ["transcribe", "--help"])
    assert result.exit_code == 0


def test_summarize_help() -> None:
    result = runner.invoke(app, ["summarize", "--help"])
    assert result.exit_code == 0


def test_run_help() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0


def test_transcribe_missing_audio(tmp_path: Path) -> None:
    result = runner.invoke(app, ["transcribe", str(tmp_path / "nope.wav")])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_summarize_missing_transcript(tmp_path: Path) -> None:
    result = runner.invoke(app, ["summarize", str(tmp_path / "nope.txt")])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_summarize_unknown_provider(tmp_path: Path) -> None:
    transcript = tmp_path / "t.txt"
    transcript.write_text("[00:00] hello\n", encoding="utf-8")
    result = runner.invoke(app, ["summarize", str(transcript), "-p", "grok"])
    assert result.exit_code == 1
    assert "provider" in result.output.lower()


def test_summarize_unknown_mode(tmp_path: Path) -> None:
    transcript = tmp_path / "t.txt"
    transcript.write_text("[00:00] hello\n", encoding="utf-8")
    result = runner.invoke(app, ["summarize", str(transcript), "-m", "ultra"])
    assert result.exit_code == 1
    assert "mode" in result.output.lower()


def test_help_does_not_load_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("provider: bad-provider\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["summarize", "--help"])
    assert result.exit_code == 0


def test_config_error_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("provider: bad-provider\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["summarize", str(tmp_path / "t.txt")])
    assert result.exit_code == 1
    assert "Configuration error" in result.output


def test_run_saves_transcript_before_llm_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"\x00" * 16)
    transcript_file = tmp_path / "out.txt"
    summary_file = tmp_path / "out_summary.txt"

    from transcript import Segment, Transcript

    fake_tr = Transcript(segments=(Segment(start=0.0, end=1.0, text="hello"),))

    import providers.llm as llm_mod
    import providers.whisper as whisper_mod

    monkeypatch.setattr(whisper_mod.WhisperTranscriber, "__init__", lambda self, model_name="large-v3": None)
    monkeypatch.setattr(whisper_mod.WhisperTranscriber, "transcribe", lambda self, audio, language="ru": fake_tr)

    def bad_summarize(self: object, text: str) -> str:
        raise ConnectionError("LLM down")

    monkeypatch.setattr(llm_mod.LLMSummarizer, "summarize", bad_summarize)

    result = runner.invoke(app, [
        "run", str(audio),
        "--transcript", str(transcript_file),
        "--summary", str(summary_file),
    ])

    assert result.exit_code == 1
    assert "LLM error" in result.output
    assert transcript_file.exists(), "transcript must be saved before LLM is called"


def test_transcribe_whisper_load_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"\x00" * 16)

    import providers.whisper as whisper_mod

    def bad_init(self: object, model_name: str = "large-v3") -> None:
        raise RuntimeError("CUDA not available")

    monkeypatch.setattr(whisper_mod.WhisperTranscriber, "__init__", bad_init)
    result = runner.invoke(app, ["transcribe", str(audio)])
    assert result.exit_code == 1
    assert "Whisper model" in result.output


def test_summarize_llm_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transcript = tmp_path / "t.txt"
    transcript.write_text("[00:00] hello world\n", encoding="utf-8")

    import providers.llm as llm_mod

    def bad_summarize(self: object, text: str) -> str:
        raise ConnectionError("LLM endpoint unreachable")

    monkeypatch.setattr(llm_mod.LLMSummarizer, "summarize", bad_summarize)
    result = runner.invoke(app, ["summarize", str(transcript)])
    assert result.exit_code == 1
    assert "LLM error" in result.output

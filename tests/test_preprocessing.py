from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config import PreprocessingSettings
from preprocessing import PreprocessingError, prepared_audio


def _settings(**kwargs) -> PreprocessingSettings:
    return PreprocessingSettings(**kwargs)


# ── Disabled ─────────────────────────────────────────────────────────────────


def test_disabled_returns_original_path(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 16)
    settings = _settings(enabled=False)
    with prepared_audio(audio, settings) as result:
        assert result == audio


def test_disabled_does_not_call_subprocess(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 16)
    settings = _settings(enabled=False)
    with patch("subprocess.run") as mock_run:
        with prepared_audio(audio, settings):
            pass
        mock_run.assert_not_called()


# ── Enabled: command structure ────────────────────────────────────────────────


def test_enabled_command_contains_expected_args(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 16)
    settings = _settings(enabled=True)

    captured: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
        captured.append(list(cmd))
        return MagicMock(returncode=0)

    with patch("subprocess.run", fake_run):
        with prepared_audio(audio, settings) as result:
            assert len(captured) == 1
            cmd = captured[0]
            assert cmd[0] == "ffmpeg"
            assert "-i" in cmd
            assert str(audio) in cmd
            assert "-ac" in cmd
            assert "1" in cmd
            assert "-ar" in cmd
            assert "16000" in cmd
            assert "-c:a" in cmd
            assert "pcm_s16le" in cmd
            assert str(result).endswith(".preprocessed.wav")


def test_enabled_no_filters_by_default(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 16)
    settings = _settings(enabled=True)

    captured: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
        captured.append(list(cmd))
        return MagicMock(returncode=0)

    with patch("subprocess.run", fake_run):
        with prepared_audio(audio, settings):
            cmd = captured[0]
            assert "-af" not in cmd


# ── Enabled: filters ─────────────────────────────────────────────────────────


def test_loudness_normalization_adds_loudnorm_filter(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 16)
    settings = _settings(enabled=True, loudness_normalization=True)

    captured: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
        captured.append(list(cmd))
        return MagicMock(returncode=0)

    with patch("subprocess.run", fake_run):
        with prepared_audio(audio, settings):
            cmd = captured[0]
            assert "-af" in cmd
            af_value = cmd[cmd.index("-af") + 1]
            assert "loudnorm" in af_value


def test_highpass_hz_adds_highpass_filter(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 16)
    settings = _settings(enabled=True, highpass_hz=70)

    captured: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
        captured.append(list(cmd))
        return MagicMock(returncode=0)

    with patch("subprocess.run", fake_run):
        with prepared_audio(audio, settings):
            cmd = captured[0]
            assert "-af" in cmd
            af_value = cmd[cmd.index("-af") + 1]
            assert "highpass" in af_value
            assert "70" in af_value


def test_both_filters_combined(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 16)
    settings = _settings(enabled=True, highpass_hz=70, loudness_normalization=True)

    captured: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
        captured.append(list(cmd))
        return MagicMock(returncode=0)

    with patch("subprocess.run", fake_run):
        with prepared_audio(audio, settings):
            cmd = captured[0]
            af_value = cmd[cmd.index("-af") + 1]
            assert "highpass" in af_value
            assert "loudnorm" in af_value


# ── Error handling ────────────────────────────────────────────────────────────


def test_ffmpeg_not_found_raises_preprocessing_error(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 16)
    settings = _settings(enabled=True)

    def fake_run(cmd: list[str], **kwargs: object) -> None:
        raise FileNotFoundError("ffmpeg")

    with patch("subprocess.run", fake_run):
        with pytest.raises(PreprocessingError, match="ffmpeg not found"):
            with prepared_audio(audio, settings):
                pass


def test_ffmpeg_failure_stderr_in_message(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 16)
    settings = _settings(enabled=True)

    def fake_run(cmd: list[str], **kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, cmd, stderr="Invalid data found")

    with patch("subprocess.run", fake_run):
        with pytest.raises(PreprocessingError, match="Invalid data found"):
            with prepared_audio(audio, settings):
                pass


# ── Temp file lifecycle ───────────────────────────────────────────────────────


def test_temp_file_deleted_after_context_keep_temp_false(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 16)
    settings = _settings(enabled=True, keep_temp=False)

    tmp_path_captured: list[Path] = []

    def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
        return MagicMock(returncode=0)

    with patch("subprocess.run", fake_run):
        with prepared_audio(audio, settings) as result:
            tmp_path_captured.append(result)

    assert not tmp_path_captured[0].exists()


def test_temp_file_kept_after_context_keep_temp_true(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 16)
    settings = _settings(enabled=True, keep_temp=True)

    tmp_path_captured: list[Path] = []

    def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
        return MagicMock(returncode=0)

    with patch("subprocess.run", fake_run):
        with prepared_audio(audio, settings) as result:
            tmp_path_captured.append(result)

    assert tmp_path_captured[0].exists()
    tmp_path_captured[0].unlink()


def test_temp_file_deleted_even_on_exception(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 16)
    settings = _settings(enabled=True, keep_temp=False)

    tmp_path_captured: list[Path] = []

    def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
        return MagicMock(returncode=0)

    with patch("subprocess.run", fake_run):
        with pytest.raises(RuntimeError):
            with prepared_audio(audio, settings) as result:
                tmp_path_captured.append(result)
                raise RuntimeError("transcription failed")

    assert not tmp_path_captured[0].exists()

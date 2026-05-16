from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from transcript import Transcript


def _make_fake_faster_whisper(captured: dict):
    class FakeInfo:
        duration = 2.0

    class FakeSegment:
        start = 0.0
        end = 1.0
        text = " hello "

    class FakeWhisperModel:
        def __init__(self, model_name: str, device: str = "cuda", compute_type: str = "float16") -> None:
            captured.update(model_name=model_name, device=device, compute_type=compute_type)

        def transcribe(
            self, audio: str, language: str, beam_size: int, vad_filter: bool, condition_on_previous_text: bool
        ):
            captured.update(
                beam_size=beam_size,
                vad_filter=vad_filter,
                condition_on_previous_text=condition_on_previous_text,
            )
            return iter([FakeSegment()]), FakeInfo()

    fake_fw = MagicMock()
    fake_fw.WhisperModel = FakeWhisperModel
    return fake_fw


@pytest.fixture
def fake_faster_whisper(monkeypatch: pytest.MonkeyPatch) -> dict:
    captured: dict = {}
    monkeypatch.setitem(sys.modules, "faster_whisper", _make_fake_faster_whisper(captured))
    return captured


def test_model_init_params(fake_faster_whisper: dict) -> None:
    from providers.whisper import WhisperTranscriber

    WhisperTranscriber(model_name="medium", device="cpu", compute_type="int8")

    assert fake_faster_whisper["model_name"] == "medium"
    assert fake_faster_whisper["device"] == "cpu"
    assert fake_faster_whisper["compute_type"] == "int8"


def test_default_compute_type_resolves_to_float16_on_cuda(fake_faster_whisper: dict) -> None:
    from providers.whisper import WhisperTranscriber

    WhisperTranscriber(model_name="large-v3", device="cuda", compute_type="default")
    assert fake_faster_whisper["compute_type"] == "float16"


def test_default_compute_type_resolves_to_int8_on_cpu(fake_faster_whisper: dict) -> None:
    from providers.whisper import WhisperTranscriber

    WhisperTranscriber(model_name="large-v3", device="cpu", compute_type="default")
    assert fake_faster_whisper["compute_type"] == "int8"


def test_cpu_without_explicit_compute_type_resolves_to_int8(fake_faster_whisper: dict) -> None:
    from providers.whisper import WhisperTranscriber

    WhisperTranscriber(model_name="large-v3", device="cpu")
    assert fake_faster_whisper["compute_type"] == "int8"


def test_default_compute_type_resolves_to_int8_on_auto(fake_faster_whisper: dict) -> None:
    from providers.whisper import WhisperTranscriber

    WhisperTranscriber(model_name="large-v3", device="auto", compute_type="default")
    assert fake_faster_whisper["compute_type"] == "int8"


def test_transcribe_params(fake_faster_whisper: dict) -> None:
    from providers.whisper import WhisperTranscriber

    tr = WhisperTranscriber(
        model_name="small",
        beam_size=3,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    result = tr.transcribe(Path("test.wav"), "en")

    assert fake_faster_whisper["beam_size"] == 3
    assert fake_faster_whisper["vad_filter"] is False
    assert fake_faster_whisper["condition_on_previous_text"] is False
    assert isinstance(result, Transcript)
    assert len(result.segments) == 1
    assert result.segments[0].text == "hello"

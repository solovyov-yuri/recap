from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from transcript import Segment, Transcript

logger = logging.getLogger(__name__)


def _set_cuda_paths() -> None:
    """Prepend venv NVIDIA lib dirs to LD_LIBRARY_PATH before CUDA libs are loaded."""
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site = Path(".venv") / "lib" / py_ver / "site-packages"
    extras = [
        str(site / "nvidia" / "cublas" / "lib"),
        str(site / "nvidia" / "cudnn" / "lib"),
    ]
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(extras + [existing])


class WhisperTranscriber:
    def __init__(self, model_name: str = "large-v3") -> None:
        _set_cuda_paths()
        from faster_whisper import WhisperModel  # noqa: PLC0415 — must be after _set_cuda_paths

        logger.info("Loading Whisper model %s…", model_name)
        self._model = WhisperModel(model_name, device="cuda", compute_type="float16")

    def transcribe(self, audio: Path, language: str = "ru") -> Transcript:
        logger.info("Transcribing %s…", audio)
        segments_iter, _ = self._model.transcribe(
            str(audio),
            language=language,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=True,
        )
        segments = [
            Segment(start=s.start, end=s.end, text=s.text.strip())
            for s in segments_iter
        ]
        return Transcript(segments=segments)

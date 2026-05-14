from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from transcript import Segment, Transcript

logger = logging.getLogger(__name__)


def _set_cuda_paths() -> None:
    """Prepend venv NVIDIA lib dirs to PATH / LD_LIBRARY_PATH before CUDA libs are loaded."""
    if sys.platform == "win32":
        site = Path(".venv") / "Lib" / "site-packages"
        lib_dir = "bin"
        env_var = "PATH"
        sep = ";"
    else:
        py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        site = Path(".venv") / "lib" / py_ver / "site-packages"
        lib_dir = "lib"
        env_var = "LD_LIBRARY_PATH"
        sep = ":"
    extras = [
        str(site / "nvidia" / "cublas" / lib_dir),
        str(site / "nvidia" / "cudnn" / lib_dir),
    ]
    existing = os.environ.get(env_var, "")
    os.environ[env_var] = sep.join(extras + [existing])


class WhisperTranscriber:
    def __init__(self, model_name: str = "large-v3") -> None:
        _set_cuda_paths()
        from faster_whisper import WhisperModel  # noqa: PLC0415
        from rich.console import Console  # noqa: PLC0415

        with Console(stderr=True).status(f"[bold cyan]Loading model {model_name}…[/]"):
            self._model = WhisperModel(model_name, device="cuda", compute_type="float16")
        logger.info("Model %s loaded.", model_name)

    def transcribe(self, audio: Path, language: str = "ru") -> Transcript:
        from rich.console import Console  # noqa: PLC0415
        from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn  # noqa: PLC0415

        logger.info("Transcribing %s…", audio)
        segments_iter, info = self._model.transcribe(
            str(audio),
            language=language,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=True,
        )
        with Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=Console(stderr=True),
        ) as progress:
            task = progress.add_task("Transcribing", total=info.duration)
            segments = []
            for s in segments_iter:
                segments.append(Segment(start=s.start, end=s.end, text=s.text.strip()))
                progress.update(task, completed=s.end)

        logger.info("Got %d segments.", len(segments))
        return Transcript(segments=segments)

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from config import PreprocessingSettings


class PreprocessingError(RuntimeError):
    pass


@contextmanager
def prepared_audio(audio: Path, settings: PreprocessingSettings) -> Iterator[Path]:
    if not settings.enabled:
        yield audio
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".preprocessed.wav") as f:
        tmp = Path(f.name)

    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(audio),
            "-ac", str(settings.channels),
            "-ar", str(settings.sample_rate),
            "-c:a", settings.codec,
        ]

        filters: list[str] = []
        if settings.highpass_hz is not None:
            filters.append(f"highpass=f={settings.highpass_hz}")
        if settings.loudness_normalization:
            filters.append(
                f"loudnorm=I={settings.target_lufs}:TP={settings.true_peak_db}:LRA={settings.loudness_range}"
            )
        if filters:
            cmd += ["-af", ",".join(filters)]

        cmd.append(str(tmp))

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError:
            raise PreprocessingError(
                "ffmpeg not found. Install ffmpeg or set preprocessing.enabled: false."
            )
        except subprocess.CalledProcessError as exc:
            raise PreprocessingError(f"ffmpeg failed: {exc.stderr.strip()}")

        yield tmp
    finally:
        if not settings.keep_temp:
            tmp.unlink(missing_ok=True)

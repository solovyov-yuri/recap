from __future__ import annotations

import tempfile
from pathlib import Path


def write_text_atomic(path: Path, text: str) -> None:
    """Write text to path atomically: write to a unique tmp file, then rename.

    Uses a unique temp filename in the same directory so concurrent writes
    to different outputs don't collide on the same .tmp sidecar.
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as f:
        tmp = Path(f.name)
        f.write(text)
    try:
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

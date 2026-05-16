from __future__ import annotations

import tempfile
from pathlib import Path


def write_text_atomic(path: Path, text: str) -> None:
    """Write text to path atomically via a unique tmp file + rename.

    A unique filename in the same directory ensures the rename is on the same
    filesystem (required for atomic replace) and avoids collisions when multiple
    outputs are written concurrently.
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

from pathlib import Path

import pytest

from utils import write_text_atomic


def test_creates_file(tmp_path: Path) -> None:
    dest = tmp_path / "out.txt"
    write_text_atomic(dest, "hello")
    assert dest.read_text(encoding="utf-8") == "hello"


def test_overwrites_existing(tmp_path: Path) -> None:
    dest = tmp_path / "out.txt"
    dest.write_text("old", encoding="utf-8")
    write_text_atomic(dest, "new")
    assert dest.read_text(encoding="utf-8") == "new"


def test_no_tmp_file_after_success(tmp_path: Path) -> None:
    dest = tmp_path / "out.txt"
    write_text_atomic(dest, "hello")
    assert not dest.with_suffix(dest.suffix + ".tmp").exists()


def test_old_file_preserved_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dest = tmp_path / "out.txt"
    dest.write_text("old content", encoding="utf-8")

    original_replace = Path.replace

    def failing_replace(self: Path, target: Path) -> Path:
        if target == dest:
            raise OSError("simulated rename failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", failing_replace)

    with pytest.raises(OSError, match="simulated"):
        write_text_atomic(dest, "new content")

    assert dest.read_text(encoding="utf-8") == "old content"
    assert not list(tmp_path.glob(f".{dest.name}.*.tmp"))

import textwrap
from pathlib import Path

import pytest

from transcript import Segment, Transcript

TIMESTAMPED = textwrap.dedent("""\
    [0.85s -> 5.22s] Привет, как дела?
    [5.22s -> 7.50s] Всё хорошо, спасибо.
""")


def test_from_file_timestamped(tmp_path: Path) -> None:
    f = tmp_path / "t.txt"
    f.write_text(TIMESTAMPED, encoding="utf-8")
    tr = Transcript.from_file(f)
    assert len(tr.segments) == 2
    assert tr.segments[0].start == pytest.approx(0.85)
    assert tr.segments[0].end == pytest.approx(5.22)
    assert tr.segments[0].text == "Привет, как дела?"
    assert tr.segments[1].text == "Всё хорошо, спасибо."


def test_from_file_no_timestamp(tmp_path: Path) -> None:
    f = tmp_path / "t.txt"
    f.write_text("просто текст\nещё строка\n", encoding="utf-8")
    tr = Transcript.from_file(f)
    assert len(tr.segments) == 2
    assert tr.segments[0].text == "просто текст"
    assert tr.segments[0].start == 0.0


def test_from_file_mixed(tmp_path: Path) -> None:
    content = "[1.00s -> 2.00s] С таймстампом\nбез таймстампа\n"
    f = tmp_path / "t.txt"
    f.write_text(content, encoding="utf-8")
    tr = Transcript.from_file(f)
    assert len(tr.segments) == 2
    assert tr.segments[0].start == 1.0
    assert tr.segments[1].start == 0.0


def test_from_file_empty_lines_ignored(tmp_path: Path) -> None:
    f = tmp_path / "t.txt"
    f.write_text("\n[1.00s -> 2.00s] Текст\n\n", encoding="utf-8")
    tr = Transcript.from_file(f)
    assert len(tr.segments) == 1


def test_to_text_strips_timestamps() -> None:
    tr = Transcript(segments=[
        Segment(0.85, 5.22, "Привет"),
        Segment(5.22, 7.50, "Пока"),
    ])
    text = tr.to_text()
    assert "0.85" not in text
    assert "5.22" not in text
    assert text == "Привет\nПока"


def test_to_text_skips_empty_segments() -> None:
    tr = Transcript(segments=[
        Segment(0.0, 0.0, ""),
        Segment(1.0, 2.0, "Текст"),
    ])
    assert tr.to_text() == "Текст"


def test_to_file_format_timestamped() -> None:
    tr = Transcript(segments=[Segment(1.0, 2.5, "Текст")])
    assert "[1.00s -> 2.50s] Текст" in tr.to_file_format()


def test_to_file_format_no_timestamp() -> None:
    tr = Transcript(segments=[Segment(0.0, 0.0, "Без времени")])
    assert "Без времени" in tr.to_file_format()
    assert "[" not in tr.to_file_format()

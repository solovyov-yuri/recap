import json

import pytest

from formatters import to_json, to_plain, to_telegram
from models import MeetingSummary


@pytest.mark.parametrize("input_text, expected", [
    # Think tags stripped
    ("<think>internal</think>result", "result"),
    # ### heading → *heading*
    ("### Заголовок\nтекст", "*Заголовок*\nтекст"),
    # ## heading
    ("## Тема\nтекст", "*Тема*\nтекст"),
    # # heading
    ("# Главное\nтекст", "*Главное*\nтекст"),
    # **bold** → *bold*
    ("**жирный** текст", "*жирный* текст"),
    # Numbered list → dashes
    ("1. первый\n2. второй", "- первый\n- второй"),
    # Bullet * item → -
    ("* элемент\n* второй", "- элемент\n- второй"),
    # Horizontal rule removed
    ("текст\n---\nещё", "текст\n\nещё"),
    # 3+ blank lines collapsed to 2
    ("а\n\n\n\nб", "а\n\nб"),
    # Existing Telegram format preserved
    ("*Тема*\n- пункт", "*Тема*\n- пункт"),
    # Mixed: think + heading
    ("<think>мысли</think>\n## Итог\n- пункт", "*Итог*\n- пункт"),
])
def test_to_telegram(input_text: str, expected: str) -> None:
    assert to_telegram(input_text) == expected


def test_to_plain_strips_think() -> None:
    assert to_plain("<think>мысли</think>ответ") == "ответ"


def test_to_plain_passthrough() -> None:
    assert to_plain("просто текст") == "просто текст"


def test_to_plain_multiline_think() -> None:
    text = "<think>\nдолгие\nрассуждения\n</think>\nрезультат"
    assert to_plain(text) == "результат"


# Special character behaviour (Telegram Markdown v1 — no backslash escaping).
# These tests document current pass-through behaviour. Unbalanced _ or ` may
# cause Telegram to reject the message; this is a known v1 limitation.

def test_to_json_structure() -> None:
    s = MeetingSummary(raw="LLM output", mode="medium")
    result = json.loads(to_json(s))
    assert result["mode"] == "medium"
    assert result["summary"] == "LLM output"


def test_to_json_strips_think_tags() -> None:
    s = MeetingSummary(raw="<think>internal</think>actual summary", mode="brief")
    result = json.loads(to_json(s))
    assert result["summary"] == "actual summary"
    assert "<think>" not in result["summary"]


def test_to_json_valid_json() -> None:
    s = MeetingSummary(raw='Привет "мир"\nновая строка', mode="detailed")
    output = to_json(s)
    parsed = json.loads(output)
    assert parsed["summary"] == 'Привет "мир"\nновая строка'


def test_to_json_preserves_unicode() -> None:
    s = MeetingSummary(raw="Итог встречи: всё решено", mode="medium")
    output = to_json(s)
    assert "Итог встречи" in output
    parsed = json.loads(output)
    assert "Итог встречи" in parsed["summary"]


@pytest.mark.parametrize("input_text, expected", [
    # Underscores in plain text pass through unchanged
    ("файл file_name.txt готов", "файл file_name.txt готов"),
    # Paired underscores pass through (Telegram renders as italic — intentional)
    ("_курсив_", "_курсив_"),
    # Backtick inline code passes through
    ("запусти `pip install recap`", "запусти `pip install recap`"),
    # Inline URL passes through (Telegram renders as hyperlink)
    ("[документация](https://example.com)", "[документация](https://example.com)"),
    # Bare brackets without URL pass through
    ("пункт [1] выполнен", "пункт [1] выполнен"),
    # Parentheses pass through
    ("(см. приложение)", "(см. приложение)"),
    # Mixed: heading with special chars in body
    ("## Итог\nфайл config_(prod).yaml", "*Итог*\nфайл config_(prod).yaml"),
])
def test_special_chars_pass_through(input_text: str, expected: str) -> None:
    assert to_telegram(input_text) == expected

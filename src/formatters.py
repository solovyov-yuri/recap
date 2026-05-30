from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models import MeetingSummary

# Output format: Telegram Markdown v1 (parse_mode="Markdown").
# Supported: *bold*, _italic_, `code`, [text](url).
#
# Known limitation: v1 has no backslash escaping. If the LLM output contains
# unbalanced _ or ` characters (e.g. snake_case identifiers, lone backticks),
# Telegram may reject the message or render it incorrectly. The prompts instruct
# the LLM to use only * for emphasis and avoid raw _, so this is rare in practice.
# If strict escaping is required in the future, migrate to MarkdownV2.


def to_telegram(text: str) -> str:
    """Convert LLM Markdown output to Telegram Markdown v1."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^#{1,6}\s+(.+)$", r"*\1*", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)
    text = re.sub(r"^\d+\.\s+", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"^\*\s+", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"^---+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def to_plain(text: str) -> str:
    """Strip think tags, otherwise pass through."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def to_json(summary: MeetingSummary) -> str:
    """Serialize a MeetingSummary to a pretty-printed JSON string."""
    import json  # noqa: PLC0415

    return json.dumps(
        {"mode": summary.mode, "summary": to_plain(summary.raw)},
        ensure_ascii=False,
        indent=2,
    )

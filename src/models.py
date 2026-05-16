from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MeetingSummary:
    raw: str
    mode: str

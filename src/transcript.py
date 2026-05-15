from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_TIMESTAMP_RE = re.compile(r"^\[(\d+\.\d+)s -> (\d+\.\d+)s\] (.+)$")


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    segments: tuple[Segment, ...]

    @classmethod
    def from_file(cls, path: Path) -> Transcript:
        segments = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            m = _TIMESTAMP_RE.match(line)
            if m:
                segments.append(Segment(
                    start=float(m.group(1)),
                    end=float(m.group(2)),
                    text=m.group(3).strip(),
                ))
            else:
                segments.append(Segment(start=0.0, end=0.0, text=line))
        return cls(segments=tuple(segments))

    def to_text(self) -> str:
        """Plain text without timestamps — for LLM input."""
        return "\n".join(s.text for s in self.segments if s.text)

    def to_file_format(self) -> str:
        """Timestamped format for writing to disk."""
        lines = []
        for s in self.segments:
            if s.start == 0.0 and s.end == 0.0:
                lines.append(s.text)
            else:
                lines.append(f"[{s.start:.2f}s -> {s.end:.2f}s] {s.text}")
        return "\n".join(lines) + "\n" if lines else ""

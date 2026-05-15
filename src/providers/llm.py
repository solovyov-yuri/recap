from __future__ import annotations

import logging

from prompts import PROMPTS, SUMMARY_PROMPT_MEDIUM_RU  # noqa: F401 — re-exported for consumers

logger = logging.getLogger(__name__)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n", 0, max_chars)
    return text[:cut if cut != -1 else max_chars]


class LLMSummarizer:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        prompt_template: str = SUMMARY_PROMPT_MEDIUM_RU,
        max_chars: int = 60_000,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._prompt_template = prompt_template
        self._max_chars = max_chars

    def summarize(self, transcript_text: str) -> str:
        import openai  # noqa: PLC0415 — deferred to keep CLI startup fast
        from rich.console import Console  # noqa: PLC0415

        if len(transcript_text) > self._max_chars:
            logger.warning(
                "Transcript truncated from %d to %d chars to fit context window.",
                len(transcript_text), self._max_chars,
            )
            transcript_text = _truncate(transcript_text, self._max_chars)
        prompt = self._prompt_template.replace("{transcript}", transcript_text)
        logger.info("Calling %s (model: %s)…", self._base_url or "openai", self._model)
        client = openai.OpenAI(
            api_key=self._api_key or ("local" if self._base_url else None),
            base_url=self._base_url,
        )
        console = Console(stderr=True)
        console.print(f"[bold cyan]Generating summary ({self._model})…[/bold cyan]")
        chunks: list[str] = []
        with client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ) as stream:
            for chunk in stream:
                if chunk.choices and (delta := chunk.choices[0].delta.content):
                    console.print(delta, end="", highlight=False, markup=False)
                    chunks.append(delta)
        console.print()
        return "".join(chunks)

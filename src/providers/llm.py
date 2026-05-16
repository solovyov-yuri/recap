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
        prompt_template: str | tuple[str, str] = PROMPTS["medium"],
        max_chars: int = 60_000,
        timeout: float = 60.0,
        max_retries: int = 2,
        retry_backoff: float = 2.0,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._prompt_template = prompt_template
        self._max_chars = max_chars
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff

    def _build_messages(self, transcript_text: str) -> list[dict[str, str]]:
        if isinstance(self._prompt_template, str):
            from prompts import SYSTEM_PROMPT_RU  # noqa: PLC0415

            system, user_template = SYSTEM_PROMPT_RU, self._prompt_template
        else:
            system, user_template = self._prompt_template
        user = user_template.replace("{transcript}", transcript_text)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def summarize(self, transcript_text: str) -> str:
        import time  # noqa: PLC0415
        import openai  # noqa: PLC0415 — deferred to keep CLI startup fast
        from rich.console import Console  # noqa: PLC0415

        _RETRYABLE = (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError)

        if len(transcript_text) > self._max_chars:
            logger.warning(
                "Transcript truncated from %d to %d chars to fit context window.",
                len(transcript_text), self._max_chars,
            )
            transcript_text = _truncate(transcript_text, self._max_chars)
        messages = self._build_messages(transcript_text)
        logger.info("Calling %s (model: %s)…", self._base_url or "openai", self._model)
        # max_retries=0: disable SDK-level retries — we manage retry logic ourselves.
        client = openai.OpenAI(
            api_key=self._api_key or ("local" if self._base_url else None),
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=0,
        )
        console = Console(stderr=True)

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                logger.warning(
                    "LLM request failed, retrying (%d/%d): %s",
                    attempt, self._max_retries, last_exc,
                )
                time.sleep(self._retry_backoff)
            try:
                console.print(f"[bold cyan]Generating summary ({self._model})…[/bold cyan]")
                chunks: list[str] = []
                # Known limitation: tokens printed to stderr during a failed attempt
                # remain visible; on retry they are printed again. The returned string
                # is always complete and correct — only the interactive display is affected.
                with client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    stream=True,
                ) as stream:
                    for chunk in stream:
                        if chunk.choices and (delta := chunk.choices[0].delta.content):
                            console.print(delta, end="", highlight=False, markup=False)
                            chunks.append(delta)
                console.print()
                return "".join(chunks)
            except _RETRYABLE as exc:
                last_exc = exc

        raise last_exc  # type: ignore[misc]

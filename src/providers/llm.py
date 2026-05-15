from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

SUMMARY_PROMPT_RU = """Ты — ассистент для подготовки саммари встреч. Отвечай только на русском языке.

Составь саммари встречи по транскрипции ниже. Выведи только текст саммари — без вводных фраз и предложений продолжить.

Структура (строго такая):
*Тема встречи*
[одно-два предложения о чём встреча]

*Ключевые обсуждения*
- [пункт]
- [пункт]

*Решения и задачи*
- [задача или решение]
- [задача или решение]

Транскрипция:
{transcript}"""


class LLMSummarizer:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        prompt_template: str = SUMMARY_PROMPT_RU,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._prompt_template = prompt_template

    def summarize(self, transcript_text: str) -> str:
        import openai  # noqa: PLC0415 — deferred to keep CLI startup fast
        from rich.console import Console  # noqa: PLC0415

        prompt = self._prompt_template.format(transcript=transcript_text)
        logger.info("Calling %s (model: %s)…", self._base_url or "openai", self._model)
        client = openai.OpenAI(api_key=self._api_key, base_url=self._base_url)
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

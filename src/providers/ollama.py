from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

SUMMARY_PROMPT_RU = """/no_think
Ты — ассистент для подготовки саммари встреч. Отвечай только на русском языке.

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


class OllamaSummarizer:
    def __init__(
        self,
        model: str = "qwen3.5:latest",
        prompt_template: str = SUMMARY_PROMPT_RU,
    ) -> None:
        self._model = model
        self._prompt_template = prompt_template

    def summarize(self, transcript_text: str) -> str:
        import ollama  # noqa: PLC0415 — deferred to keep CLI startup fast
        from rich.console import Console  # noqa: PLC0415

        prompt = self._prompt_template.format(transcript=transcript_text)
        logger.info("Calling Ollama model %s…", self._model)
        with Console(stderr=True).status(f"[bold cyan]Generating summary ({self._model})…[/]"):
            response = ollama.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
            )
        return response.message.content

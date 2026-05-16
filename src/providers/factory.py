from __future__ import annotations

from config import PROVIDER_PRESETS, Settings


def make_summarizer(
    settings: Settings,
    provider_name: str,
    mode_name: str,
    model_override: str | None = None,
    summary_language: str | None = None,
) -> "LLMSummarizer":
    """Validate provider, mode, and language, then build an LLMSummarizer.

    Raises ValueError for unknown provider, mode, or language.
    """
    from prompts import CHUNK_PROMPTS, PROMPTS, get_prompt  # noqa: PLC0415

    if provider_name not in PROVIDER_PRESETS:
        available = ", ".join(PROVIDER_PRESETS)
        raise ValueError(f"Unknown provider: {provider_name!r}. Available: {available}")

    # Default to "ru" — the only language with prompt content shipped.
    # transcription_language is NOT inherited here so that -l en (English audio)
    # still produces a Russian summary without requiring --summary-language ru.
    effective_lang = summary_language or settings.summary_language or "ru"
    try:
        prompt_template = get_prompt(effective_lang, mode_name)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc

    chunk_prompt = CHUNK_PROMPTS.get(effective_lang, CHUNK_PROMPTS["ru"])

    from providers.llm import LLMSummarizer  # noqa: PLC0415

    return LLMSummarizer(
        model=model_override or settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url or PROVIDER_PRESETS[provider_name],
        max_chars=settings.max_transcript_chars,
        prompt_template=prompt_template,
        chunk_prompt=chunk_prompt,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_retries,
        chunking_mode=settings.chunking_mode,
    )


def make_transcriber(settings: Settings) -> "WhisperTranscriber":
    """Build a WhisperTranscriber from settings."""
    from providers.whisper import WhisperTranscriber  # noqa: PLC0415

    return WhisperTranscriber(
        model_name=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        beam_size=settings.whisper_beam_size,
        vad_filter=settings.whisper_vad_filter,
        condition_on_previous_text=settings.whisper_condition_on_previous_text,
    )

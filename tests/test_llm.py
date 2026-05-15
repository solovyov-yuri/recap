from prompts import PROMPTS, SUMMARY_PROMPT_BRIEF_RU, SUMMARY_PROMPT_DETAILED_RU, SUMMARY_PROMPT_MEDIUM_RU
from providers.llm import LLMSummarizer


def test_prompts_has_all_modes() -> None:
    assert set(PROMPTS) == {"brief", "medium", "detailed"}


def test_all_prompts_contain_placeholder() -> None:
    for name, template in PROMPTS.items():
        assert "{transcript}" in template, f"Mode {name!r} missing {{transcript}} placeholder"


def test_default_prompt_is_medium() -> None:
    s = LLMSummarizer(model="test")
    assert s._prompt_template is SUMMARY_PROMPT_MEDIUM_RU


def test_prompts_map_to_correct_constants() -> None:
    assert PROMPTS["brief"] is SUMMARY_PROMPT_BRIEF_RU
    assert PROMPTS["medium"] is SUMMARY_PROMPT_MEDIUM_RU
    assert PROMPTS["detailed"] is SUMMARY_PROMPT_DETAILED_RU

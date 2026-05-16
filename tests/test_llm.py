import pytest
from prompts import PROMPTS, SUMMARY_PROMPT_BRIEF_RU, SUMMARY_PROMPT_DETAILED_RU, SUMMARY_PROMPT_MEDIUM_RU
from providers.llm import LLMSummarizer


def test_prompts_has_all_modes() -> None:
    assert set(PROMPTS) == {"brief", "medium", "detailed"}


def test_all_prompts_contain_placeholder() -> None:
    for name, (_, user_template) in PROMPTS.items():
        assert "{transcript}" in user_template, f"Mode {name!r} missing {{transcript}} placeholder"


def test_default_prompt_is_medium() -> None:
    s = LLMSummarizer(model="test")
    assert s._prompt_template is PROMPTS["medium"]


def test_prompts_map_to_correct_constants() -> None:
    assert PROMPTS["brief"][1] is SUMMARY_PROMPT_BRIEF_RU
    assert PROMPTS["medium"][1] is SUMMARY_PROMPT_MEDIUM_RU
    assert PROMPTS["detailed"][1] is SUMMARY_PROMPT_DETAILED_RU


def test_transcript_only_in_user_template() -> None:
    for name, (system, user_template) in PROMPTS.items():
        assert "{transcript}" not in system, f"Mode {name!r}: placeholder leaked into system prompt"
        assert "{transcript}" in user_template, f"Mode {name!r}: placeholder missing from user template"


def test_user_templates_have_transcript_delimiters() -> None:
    for name, (_, user_template) in PROMPTS.items():
        assert "<transcript>" in user_template, f"Mode {name!r}: missing <transcript> tag"
        assert "</transcript>" in user_template, f"Mode {name!r}: missing </transcript> tag"


def test_system_prompt_has_injection_warning() -> None:
    for name, (system, _) in PROMPTS.items():
        assert "transcript" in system.lower(), f"Mode {name!r}: system prompt doesn't mention transcript boundary"


def test_all_prompts_share_system_prompt() -> None:
    system_prompts = {system for system, _ in PROMPTS.values()}
    assert len(system_prompts) == 1, "All modes should use the same system prompt"


def test_build_messages_structure() -> None:
    summarizer = LLMSummarizer(model="test")
    messages = summarizer._build_messages("hello world")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_build_messages_transcript_in_user_only() -> None:
    summarizer = LLMSummarizer(model="test")
    messages = summarizer._build_messages("secret meeting content")
    assert "secret meeting content" in messages[1]["content"]
    assert "secret meeting content" not in messages[0]["content"]


def test_build_messages_transcript_wrapped_in_delimiters() -> None:
    summarizer = LLMSummarizer(model="test")
    messages = summarizer._build_messages("my transcript")
    user_content = messages[1]["content"]
    assert "<transcript>" in user_content
    assert "my transcript" in user_content
    assert "</transcript>" in user_content


def test_build_messages_str_template_uses_system_prompt() -> None:
    summarizer = LLMSummarizer(model="test", prompt_template="Task: {transcript}")
    messages = summarizer._build_messages("hello")
    assert messages[0]["role"] == "system"
    assert len(messages[0]["content"]) > 0
    assert "hello" in messages[1]["content"]


def test_summarize_passes_messages_to_client(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict] = []

    class FakeChunk:
        choices = [type("C", (), {"delta": type("D", (), {"content": "ok"})()})()]

    class FakeStream:
        def __enter__(self) -> "FakeStream":
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def __iter__(self):
            yield FakeChunk()

    class FakeCompletions:
        def create(self, model: str, messages: list, stream: bool) -> FakeStream:
            captured.extend(messages)
            return FakeStream()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr("openai.OpenAI", lambda **kw: FakeClient())

    summarizer = LLMSummarizer(model="test", base_url="http://localhost:1234/v1")
    summarizer.summarize("test content")

    assert len(captured) == 2
    assert captured[0]["role"] == "system"
    assert captured[1]["role"] == "user"
    assert "test content" in captured[1]["content"]
    assert "test content" not in captured[0]["content"]

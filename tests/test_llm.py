import pytest
from unittest.mock import MagicMock
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


def _make_fake_openai(response_chunks: list[str], *, captured_kwargs: dict | None = None):
    """Return a fake openai.OpenAI class that streams the given response chunks."""

    class FakeChunk:
        def __init__(self, text: str) -> None:
            self.choices = [type("C", (), {"delta": type("D", (), {"content": text})()})()]

    class FakeStream:
        def __init__(self, chunks: list[str]) -> None:
            self._chunks = chunks

        def __enter__(self) -> "FakeStream":
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def __iter__(self):
            for text in self._chunks:
                yield FakeChunk(text)

    class FakeCompletions:
        def create(self, model: str, messages: list, stream: bool, **kwargs) -> FakeStream:
            if captured_kwargs is not None:
                captured_kwargs.update(kwargs)
            return FakeStream(response_chunks)

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    def fake_openai_cls(**kw) -> FakeClient:
        if captured_kwargs is not None:
            captured_kwargs.update(kw)
        return FakeClient()

    return fake_openai_cls


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
        def create(self, model: str, messages: list, stream: bool, **kwargs) -> FakeStream:
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


def test_timeout_passed_to_openai_client(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr("openai.OpenAI", _make_fake_openai(["ok"], captured_kwargs=captured))
    monkeypatch.setattr("time.sleep", lambda _: None)

    summarizer = LLMSummarizer(model="test", base_url="http://localhost:1234/v1", timeout=42.0)
    summarizer.summarize("hi")

    assert captured.get("timeout") == 42.0


def test_retry_succeeds_on_second_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai

    call_count = [0]

    class FakeChunk:
        choices = [type("C", (), {"delta": type("D", (), {"content": "result"})()})()]

    class FakeStream:
        def __enter__(self) -> "FakeStream":
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def __iter__(self):
            yield FakeChunk()

    class FakeCompletions:
        def create(self, **kw) -> FakeStream:
            call_count[0] += 1
            if call_count[0] == 1:
                raise openai.APITimeoutError(request=MagicMock())
            return FakeStream()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr("openai.OpenAI", lambda **kw: FakeClient())
    monkeypatch.setattr("time.sleep", lambda _: None)

    summarizer = LLMSummarizer(model="test", base_url="http://localhost:1234/v1", max_retries=1, retry_backoff=0)
    result = summarizer.summarize("content")

    assert result == "result"
    assert call_count[0] == 2


def test_retry_exhausted_raises_last_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai

    class FakeCompletions:
        def create(self, **kw) -> None:
            raise openai.APIConnectionError(request=MagicMock())

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr("openai.OpenAI", lambda **kw: FakeClient())
    monkeypatch.setattr("time.sleep", lambda _: None)

    summarizer = LLMSummarizer(model="test", base_url="http://localhost:1234/v1", max_retries=2, retry_backoff=0)
    with pytest.raises(openai.APIConnectionError):
        summarizer.summarize("content")


def test_non_retryable_error_propagates_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = [0]

    class FakeCompletions:
        def create(self, **kw) -> None:
            call_count[0] += 1
            raise ValueError("bad input")

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr("openai.OpenAI", lambda **kw: FakeClient())
    monkeypatch.setattr("time.sleep", lambda _: None)

    summarizer = LLMSummarizer(model="test", base_url="http://localhost:1234/v1", max_retries=2, retry_backoff=0)
    with pytest.raises(ValueError, match="bad input"):
        summarizer.summarize("content")

    assert call_count[0] == 1

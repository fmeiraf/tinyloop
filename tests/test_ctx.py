"""
Tests for the CTX (Context Analyzer) feature.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from tinyloop.ctx import (
    CTXAnalyzer,
    CTXMiddleware,
    CTXResult,
    CTXStatus,
    CTXThresholdExceeded,
    TiktokenTokenizer,
    TokenCategory,
    analyze_conversation,
    categorize_message,
    categorize_messages,
    create_ctx_hook,
    get_status,
    get_tokenizer,
    is_anthropic_model,
    is_claude_3_or_later,
    is_openai_model,
)


# =============================================================================
# Tokenizer Tests
# =============================================================================


class TestModelDetection:
    """Test model detection functions."""

    def test_is_anthropic_model(self):
        assert is_anthropic_model("claude-3-sonnet-20240229")
        assert is_anthropic_model("anthropic/claude-sonnet-4-20250514")
        assert is_anthropic_model("Claude-3-Opus")
        assert not is_anthropic_model("gpt-4")
        assert not is_anthropic_model("openai/gpt-4-turbo")

    def test_is_claude_3_or_later(self):
        assert is_claude_3_or_later("claude-3-sonnet-20240229")
        assert is_claude_3_or_later("claude-3-5-sonnet-20241022")
        assert is_claude_3_or_later("claude-sonnet-4-20250514")
        assert is_claude_3_or_later("claude-opus-4-20250514")
        assert not is_claude_3_or_later("claude-2")
        assert not is_claude_3_or_later("claude-instant-1.2")

    def test_is_openai_model(self):
        assert is_openai_model("gpt-4")
        assert is_openai_model("gpt-3.5-turbo")
        assert is_openai_model("openai/gpt-4-turbo")
        assert is_openai_model("o1-preview")
        assert is_openai_model("o3-mini")
        assert not is_openai_model("claude-3-sonnet")
        assert not is_openai_model("llama-2-70b")


class TestTiktokenTokenizer:
    """Test tiktoken-based tokenizer."""

    def test_count_tokens_basic(self):
        tokenizer = TiktokenTokenizer()
        count = tokenizer.count_tokens("Hello, world!")
        assert count > 0
        assert count < 10  # Should be ~3-4 tokens

    def test_count_tokens_empty(self):
        tokenizer = TiktokenTokenizer()
        count = tokenizer.count_tokens("")
        assert count == 0

    def test_count_message_tokens(self):
        tokenizer = TiktokenTokenizer()
        message = {"role": "user", "content": "Hello, how are you?"}
        count = tokenizer.count_message_tokens(message)
        assert count > 0

    def test_count_message_with_tool_calls(self):
        tokenizer = TiktokenTokenizer()
        message = {
            "role": "assistant",
            "content": "Let me check the weather.",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "San Francisco"}',
                    },
                }
            ],
        }
        count = tokenizer.count_message_tokens(message)
        assert count > 10  # Should include tool call overhead

    def test_count_messages_tokens(self):
        tokenizer = TiktokenTokenizer()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        count = tokenizer.count_messages_tokens(messages)
        assert count > 0

    def test_tokenizer_name(self):
        tokenizer = TiktokenTokenizer()
        assert "tiktoken" in tokenizer.name
        assert "cl100k_base" in tokenizer.name


class TestGetTokenizer:
    """Test tokenizer factory function."""

    def test_get_tokenizer_default(self):
        tokenizer = get_tokenizer()
        assert isinstance(tokenizer, TiktokenTokenizer)
        assert "cl100k_base" in tokenizer.name

    def test_get_tokenizer_openai(self):
        tokenizer = get_tokenizer("gpt-4")
        assert isinstance(tokenizer, TiktokenTokenizer)

    def test_get_tokenizer_anthropic_offline(self):
        tokenizer = get_tokenizer("anthropic/claude-sonnet-4-20250514", offline=True)
        assert isinstance(tokenizer, TiktokenTokenizer)

    def test_get_tokenizer_unknown_model(self):
        tokenizer = get_tokenizer("some-unknown-model")
        assert isinstance(tokenizer, TiktokenTokenizer)
        assert "cl100k_base" in tokenizer.name


# =============================================================================
# Category Tests
# =============================================================================


class TestCategorization:
    """Test token categorization."""

    @pytest.fixture
    def tokenizer(self):
        return TiktokenTokenizer()

    def test_categorize_system_message(self, tokenizer):
        message = {"role": "system", "content": "You are a helpful assistant."}
        category, tokens, breakdown = categorize_message(message, tokenizer)
        assert category == TokenCategory.SYSTEM
        assert tokens > 0
        assert breakdown is None

    def test_categorize_user_message(self, tokenizer):
        message = {"role": "user", "content": "Hello, world!"}
        category, tokens, breakdown = categorize_message(message, tokenizer)
        assert category == TokenCategory.USER
        assert tokens > 0

    def test_categorize_assistant_message(self, tokenizer):
        message = {"role": "assistant", "content": "Hello! How can I help?"}
        category, tokens, breakdown = categorize_message(message, tokenizer)
        assert category == TokenCategory.ASSISTANT
        assert tokens > 0

    def test_categorize_tool_message(self, tokenizer):
        message = {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": '{"temperature": 72}',
        }
        category, tokens, breakdown = categorize_message(message, tokenizer)
        assert category == TokenCategory.TOOLS
        assert tokens > 0

    def test_categorize_messages(self, tokenizer):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        categories = categorize_messages(messages, tokenizer)
        assert categories.system > 0
        assert categories.user > 0
        assert categories.assistant > 0
        assert categories.total() == sum(
            [categories.system, categories.user, categories.assistant, categories.tools]
        )


# =============================================================================
# Analyzer Tests
# =============================================================================


class TestCTXAnalyzer:
    """Test CTX analyzer."""

    @pytest.fixture
    def simple_conversation(self):
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help you today?"},
        ]

    @pytest.fixture
    def long_conversation(self):
        """Create a conversation that will exceed the threshold."""
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        # Add many messages to exceed threshold
        for i in range(100):
            messages.append({"role": "user", "content": f"This is message {i}. " * 50})
            messages.append(
                {
                    "role": "assistant",
                    "content": f"This is response {i}. " * 100,
                }
            )
        return messages

    def test_analyzer_initialization(self):
        analyzer = CTXAnalyzer()
        assert analyzer.context_window == 168_000
        assert analyzer.threshold == 0.4
        assert analyzer.threshold_tokens == int(168_000 * 0.4)

    def test_analyzer_custom_params(self):
        analyzer = CTXAnalyzer(
            model="gpt-4",
            context_window=100_000,
            threshold=0.5,
        )
        assert analyzer.context_window == 100_000
        assert analyzer.threshold == 0.5
        assert analyzer.threshold_tokens == 50_000

    def test_analyze_simple_conversation(self, simple_conversation):
        analyzer = CTXAnalyzer(context_window=168_000, threshold=0.4)
        result = analyzer.analyze(simple_conversation)

        assert isinstance(result, CTXResult)
        assert result.total_tokens > 0
        assert result.is_in_dumb_zone is False
        assert result.percentage_used < 0.4
        assert len(result.rounds) == len(simple_conversation)

    def test_analyze_returns_rounds(self, simple_conversation):
        analyzer = CTXAnalyzer()
        result = analyzer.analyze(simple_conversation)

        assert len(result.rounds) == 3
        assert result.rounds[0].role == "system"
        assert result.rounds[1].role == "user"
        assert result.rounds[2].role == "assistant"

        # Cumulative tokens should increase
        for i in range(1, len(result.rounds)):
            assert (
                result.rounds[i].cumulative_tokens > result.rounds[i - 1].cumulative_tokens
            )

    def test_analyze_detects_dumb_zone(self, long_conversation):
        # Use a small context window to trigger dumb zone
        analyzer = CTXAnalyzer(context_window=1000, threshold=0.4)
        result = analyzer.analyze(long_conversation)

        assert result.is_in_dumb_zone is True
        assert result.dumb_zone_round is not None
        assert result.dumb_zone_round > 0

    def test_status_simple_conversation(self, simple_conversation):
        analyzer = CTXAnalyzer()
        status = analyzer.status(simple_conversation)

        assert isinstance(status, CTXStatus)
        assert status.total_tokens > 0
        assert status.is_in_dumb_zone is False
        assert status.tokens_until_dumb_zone > 0
        assert status.message != ""

    def test_status_message_format(self, simple_conversation):
        analyzer = CTXAnalyzer()
        status = analyzer.status(simple_conversation)

        # Safe status should not have warning indicators
        assert "DUMB ZONE" not in status.message

    def test_analyze_with_tools(self, simple_conversation):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]
        analyzer = CTXAnalyzer()
        result = analyzer.analyze(simple_conversation, tools=tools)

        assert result.categories["tools"] > 0


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.fixture
    def simple_messages(self):
        return [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi!"},
        ]

    def test_analyze_conversation(self, simple_messages):
        result = analyze_conversation(simple_messages)
        assert isinstance(result, CTXResult)
        assert result.total_tokens > 0

    def test_get_status(self, simple_messages):
        status = get_status(simple_messages)
        assert isinstance(status, CTXStatus)
        assert status.total_tokens > 0


# =============================================================================
# Middleware Tests
# =============================================================================


class TestCTXMiddleware:
    """Test CTX middleware."""

    @pytest.fixture
    def simple_messages(self):
        return [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]

    @pytest.fixture
    def mock_llm(self, simple_messages):
        """Create a mock LLM-like object."""

        class MockLLM:
            def __init__(self, messages):
                self._history = messages
                self.model = "gpt-4"

            def get_history(self):
                return self._history

        return MockLLM(simple_messages)

    def test_middleware_initialization(self):
        middleware = CTXMiddleware()
        assert middleware.context_window == 168_000
        assert middleware.threshold == 0.4
        assert middleware.action == "warn"

    def test_middleware_check_with_llm(self, mock_llm):
        middleware = CTXMiddleware()
        status = middleware.check(mock_llm)

        assert isinstance(status, CTXStatus)
        assert status.total_tokens > 0
        assert status.is_in_dumb_zone is False

    def test_middleware_check_with_messages(self, simple_messages):
        middleware = CTXMiddleware()
        status = middleware.check(messages=simple_messages)

        assert isinstance(status, CTXStatus)
        assert status.total_tokens > 0

    def test_middleware_analyze(self, mock_llm):
        middleware = CTXMiddleware()
        result = middleware.analyze(mock_llm)

        assert isinstance(result, CTXResult)
        assert result.total_tokens > 0

    def test_middleware_raises_on_threshold(self):
        middleware = CTXMiddleware(
            context_window=100,  # Very small
            threshold=0.1,  # 10 tokens
            action="raise",
        )
        messages = [
            {"role": "user", "content": "Hello world! " * 50}  # Many tokens
        ]

        with pytest.raises(CTXThresholdExceeded) as exc_info:
            middleware.check(messages=messages)

        assert exc_info.value.total_tokens > 10
        assert exc_info.value.threshold_tokens == 10

    def test_middleware_warns_on_threshold(self, capsys):
        middleware = CTXMiddleware(
            context_window=100,
            threshold=0.1,
            action="warn",
        )
        messages = [{"role": "user", "content": "Hello world! " * 50}]

        status = middleware.check(messages=messages)

        assert status.is_in_dumb_zone is True
        # Warning should be printed to stderr
        captured = capsys.readouterr()
        assert "CTX WARNING" in captured.err or "Dumb zone" in captured.err

    def test_middleware_custom_warning_callback(self, simple_messages):
        warning_called = []

        def custom_warning(status):
            warning_called.append(status)

        middleware = CTXMiddleware(
            context_window=10,  # Very small
            threshold=0.1,
            action="warn",
            on_warning=custom_warning,
        )
        messages = [{"role": "user", "content": "Hello world! " * 50}]

        middleware.check(messages=messages)

        assert len(warning_called) == 1
        assert warning_called[0].is_in_dumb_zone is True

    def test_middleware_get_last_status(self, mock_llm):
        middleware = CTXMiddleware()

        # Before check, should be None
        assert middleware.get_status() is None

        # After check, should have status
        middleware.check(mock_llm)
        assert middleware.get_status() is not None

    def test_create_ctx_hook(self, simple_messages):
        check_ctx = create_ctx_hook(threshold=0.4)
        status = check_ctx(messages=simple_messages)

        assert isinstance(status, CTXStatus)
        assert status.total_tokens > 0


# =============================================================================
# CLI Tests
# =============================================================================


class TestCLI:
    """Test CLI functionality."""

    @pytest.fixture
    def conversation_file(self, tmp_path):
        """Create a temporary conversation file."""
        data = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        }
        file_path = tmp_path / "conversation.json"
        with open(file_path, "w") as f:
            json.dump(data, f)
        return file_path

    @pytest.fixture
    def minimal_conversation_file(self, tmp_path):
        """Create a minimal conversation file (just messages array)."""
        data = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi!"},
        ]
        file_path = tmp_path / "minimal.json"
        with open(file_path, "w") as f:
            json.dump(data, f)
        return file_path

    def test_load_conversation_from_file(self, conversation_file):
        from tinyloop.ctx.cli import load_conversation

        data = load_conversation(conversation_file)
        assert "messages" in data
        assert len(data["messages"]) == 3

    def test_load_conversation_minimal_format(self, minimal_conversation_file):
        from tinyloop.ctx.cli import load_conversation

        data = load_conversation(minimal_conversation_file)
        assert "messages" in data
        assert len(data["messages"]) == 2

    def test_format_simple_output_safe(self):
        from tinyloop.ctx.cli import format_simple_output
        from tinyloop.ctx import TokenCategories

        result = CTXResult(
            model="gpt-4",
            context_window=168_000,
            threshold=0.4,
            threshold_tokens=67_200,
            total_tokens=1000,
            percentage_used=0.006,
            is_in_dumb_zone=False,
            dumb_zone_round=None,
            categories={"system": 100, "user": 400, "assistant": 500, "tools": 0},
            categories_detail=TokenCategories(system=100, user=400, assistant=500, tools=0),
            rounds=[],
            tokenizer_used="tiktoken (cl100k_base)",
        )

        output = format_simple_output(result)
        assert "1,000" in output
        assert "67,200" in output
        assert "DUMB ZONE" not in output

    def test_format_simple_output_dumb_zone(self):
        from tinyloop.ctx.cli import format_simple_output
        from tinyloop.ctx import TokenCategories

        result = CTXResult(
            model="gpt-4",
            context_window=168_000,
            threshold=0.4,
            threshold_tokens=67_200,
            total_tokens=70_000,
            percentage_used=0.417,
            is_in_dumb_zone=True,
            dumb_zone_round=50,
            categories={"system": 100, "user": 30000, "assistant": 39900, "tools": 0},
            categories_detail=TokenCategories(system=100, user=30000, assistant=39900, tools=0),
            rounds=[],
            tokenizer_used="tiktoken (cl100k_base)",
        )

        output = format_simple_output(result)
        assert "70,000" in output
        assert "DUMB ZONE" in output


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.integration
class TestIntegration:
    """Integration tests that may use real APIs."""

    @pytest.fixture
    def anthropic_available(self):
        return os.getenv("ANTHROPIC_API_KEY") is not None

    def test_anthropic_tokenizer_with_api(self, anthropic_available):
        """Test Anthropic API tokenizer if API key is available."""
        if not anthropic_available:
            pytest.skip("ANTHROPIC_API_KEY not available")

        from tinyloop.ctx import AnthropicAPITokenizer

        tokenizer = AnthropicAPITokenizer("claude-sonnet-4-20250514")
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        count = tokenizer.count_messages_tokens(messages)
        assert count > 0

    def test_full_analysis_with_anthropic(self, anthropic_available):
        """Test full analysis with Anthropic model."""
        if not anthropic_available:
            pytest.skip("ANTHROPIC_API_KEY not available")

        analyzer = CTXAnalyzer(
            model="anthropic/claude-sonnet-4-20250514",
            offline=False,
        )
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help?"},
        ]

        result = analyzer.analyze(messages)
        assert result.total_tokens > 0
        assert "Anthropic" in result.tokenizer_used

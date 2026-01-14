"""
Tokenizer abstraction layer for CTX analysis.

Provides tokenizers for different LLM providers:
- Anthropic: Uses official messages.count_tokens() API
- OpenAI: Uses tiktoken with model-specific encodings
- Fallback: Uses tiktoken with cl100k_base encoding
"""

from abc import ABC, abstractmethod
from typing import Any

import tiktoken


class Tokenizer(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        pass

    @abstractmethod
    def count_message_tokens(self, message: dict[str, Any]) -> int:
        """Count tokens in a message dict."""
        pass

    @abstractmethod
    def count_messages_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens in a list of messages."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the tokenizer name for reporting."""
        pass


class TiktokenTokenizer(Tokenizer):
    """Tokenizer using tiktoken library."""

    def __init__(self, model: str | None = None, encoding: str = "cl100k_base"):
        """
        Initialize tiktoken tokenizer.

        Args:
            model: Model name to get encoding for (e.g., "gpt-4")
            encoding: Fallback encoding name if model not found
        """
        self._model = model
        self._encoding_name = encoding

        if model:
            try:
                self._encoding = tiktoken.encoding_for_model(model)
                self._encoding_name = self._encoding.name
            except KeyError:
                self._encoding = tiktoken.get_encoding(encoding)
        else:
            self._encoding = tiktoken.get_encoding(encoding)

    @property
    def name(self) -> str:
        return f"tiktoken ({self._encoding_name})"

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def count_message_tokens(self, message: dict[str, Any]) -> int:
        """
        Count tokens in a message dict.

        Handles:
        - Simple text content
        - Multi-modal content (text + images)
        - Tool calls
        - Tool responses
        """
        tokens = 0

        # Count role tokens (approximately 1-2 tokens per role marker)
        tokens += 4  # Base overhead per message

        content = message.get("content")
        if content:
            if isinstance(content, str):
                tokens += self.count_tokens(content)
            elif isinstance(content, list):
                # Multi-modal content
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            tokens += self.count_tokens(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            # Image token estimation
                            tokens += self._estimate_image_tokens(item)

        # Handle tool calls in assistant messages
        tool_calls = message.get("tool_calls", [])
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                func = tool_call.get("function", {})
                tokens += self.count_tokens(func.get("name", ""))
                tokens += self.count_tokens(func.get("arguments", ""))
                tokens += 10  # Overhead for tool call structure

        # Handle tool response
        if message.get("role") == "tool":
            tool_call_id = message.get("tool_call_id", "")
            if tool_call_id:
                tokens += self.count_tokens(tool_call_id)
            name = message.get("name", "")
            if name:
                tokens += self.count_tokens(name)

        return tokens

    def _estimate_image_tokens(self, image_item: dict[str, Any]) -> int:
        """
        Estimate tokens for an image.

        For OpenAI models, this depends on image size and detail level.
        Default estimate: 1000 tokens per image.
        """
        # OpenAI's image token calculation depends on dimensions
        # For simplicity, use a conservative estimate
        # High detail: up to 765 tokens per tile
        # Low detail: 85 tokens
        image_url = image_item.get("image_url", {})
        detail = image_url.get("detail", "auto")

        if detail == "low":
            return 85
        else:
            # High/auto detail - use conservative estimate
            return 1000

    def count_messages_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens in a list of messages."""
        total = 0
        for message in messages:
            total += self.count_message_tokens(message)
        # Add tokens for the conversation structure
        total += 3  # Every reply is primed with <|start|>assistant<|message|>
        return total


class AnthropicAPITokenizer(Tokenizer):
    """
    Tokenizer using Anthropic's official count_tokens API.

    Falls back to tiktoken estimation if API is unavailable.
    """

    def __init__(self, model: str):
        """
        Initialize Anthropic API tokenizer.

        Args:
            model: The Claude model name (e.g., "claude-sonnet-4-20250514")
        """
        self._model = model
        self._client = None
        self._fallback = TiktokenTokenizer(encoding="cl100k_base")
        self._api_available = False

        # Try to initialize Anthropic client
        try:
            import anthropic

            self._client = anthropic.Anthropic()
            self._api_available = True
        except ImportError:
            pass
        except anthropic.AuthenticationError:
            pass

    @property
    def name(self) -> str:
        if self._api_available:
            return "Anthropic API"
        return f"tiktoken (cl100k_base) [Anthropic API unavailable]"

    def _normalize_model_name(self, model: str) -> str:
        """
        Normalize model name for Anthropic API.

        Removes provider prefix if present (e.g., "anthropic/claude-..." -> "claude-...")
        """
        if "/" in model:
            return model.split("/", 1)[1]
        return model

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string using fallback tokenizer."""
        return self._fallback.count_tokens(text)

    def count_message_tokens(self, message: dict[str, Any]) -> int:
        """
        Count tokens in a single message.

        Uses Anthropic API if available, otherwise falls back to tiktoken.
        """
        if not self._api_available:
            return self._fallback.count_message_tokens(message)

        # For single message, wrap it and call count_messages_tokens
        return self._fallback.count_message_tokens(message)

    def count_messages_tokens(self, messages: list[dict[str, Any]]) -> int:
        """
        Count tokens in a list of messages using Anthropic API.

        Uses the official messages.count_tokens() endpoint.
        """
        if not self._api_available or not self._client:
            return self._fallback.count_messages_tokens(messages)

        try:
            # Separate system message from other messages
            system_message = None
            other_messages = []

            for msg in messages:
                if msg.get("role") == "system":
                    # Anthropic expects system as a string, not a message
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        # Extract text from multi-part content
                        content = " ".join(
                            item.get("text", "")
                            for item in content
                            if item.get("type") == "text"
                        )
                    system_message = content
                else:
                    # Convert OpenAI-style messages to Anthropic format
                    converted = self._convert_message_to_anthropic(msg)
                    if converted:
                        other_messages.append(converted)

            # Ensure messages alternate between user and assistant
            other_messages = self._ensure_alternating_messages(other_messages)

            if not other_messages:
                # If no messages, just count the system message
                if system_message:
                    return self._fallback.count_tokens(system_message)
                return 0

            # Call Anthropic API
            model_name = self._normalize_model_name(self._model)
            count_result = self._client.messages.count_tokens(
                model=model_name,
                messages=other_messages,
                system=system_message or "",
            )

            return count_result.input_tokens

        except Exception:
            # Fall back to tiktoken on any API error
            return self._fallback.count_messages_tokens(messages)

    def _convert_message_to_anthropic(
        self, message: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Convert an OpenAI-style message to Anthropic format."""
        role = message.get("role")
        content = message.get("content")

        # Skip tool messages - they need special handling
        if role == "tool":
            # Convert tool response to user message with tool_result
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.get("tool_call_id", ""),
                        "content": content or "",
                    }
                ],
            }

        if role not in ("user", "assistant"):
            return None

        # Handle assistant messages with tool calls
        if role == "assistant" and message.get("tool_calls"):
            content_blocks = []
            if content:
                content_blocks.append({"type": "text", "text": content})

            for tool_call in message.get("tool_calls", []):
                func = tool_call.get("function", {})
                import json

                try:
                    args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}

                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.get("id", ""),
                        "name": func.get("name", ""),
                        "input": args,
                    }
                )

            return {"role": "assistant", "content": content_blocks}

        # Handle regular messages
        if isinstance(content, str):
            return {"role": role, "content": content}
        elif isinstance(content, list):
            # Convert multi-modal content
            anthropic_content = []
            for item in content:
                if item.get("type") == "text":
                    anthropic_content.append(
                        {"type": "text", "text": item.get("text", "")}
                    )
                elif item.get("type") == "image_url":
                    # Convert image to Anthropic format
                    image_url = item.get("image_url", {})
                    url = image_url.get("url", "")
                    if url.startswith("data:"):
                        # Parse data URI
                        media_type, _, data = url[5:].partition(";base64,")
                        anthropic_content.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": data,
                                },
                            }
                        )
            return {"role": role, "content": anthropic_content}

        return {"role": role, "content": content or ""}

    def _ensure_alternating_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Ensure messages alternate between user and assistant.

        Anthropic API requires alternating roles starting with user.
        """
        if not messages:
            return messages

        result = []
        last_role = None

        for msg in messages:
            role = msg.get("role")

            # If same role as previous, merge or skip
            if role == last_role:
                if role == "user" and result:
                    # Merge user messages
                    last_content = result[-1].get("content", "")
                    curr_content = msg.get("content", "")
                    if isinstance(last_content, str) and isinstance(curr_content, str):
                        result[-1]["content"] = last_content + "\n" + curr_content
                    elif isinstance(last_content, list) and isinstance(
                        curr_content, list
                    ):
                        result[-1]["content"] = last_content + curr_content
                continue

            # First message must be user
            if not result and role != "user":
                result.append({"role": "user", "content": "(conversation continues)"})

            result.append(msg)
            last_role = role

        return result


def is_anthropic_model(model: str) -> bool:
    """Check if model is from Anthropic."""
    model_lower = model.lower()
    return any(x in model_lower for x in ["claude", "anthropic"])


def is_claude_3_or_later(model: str) -> bool:
    """
    Check if model is Claude 3 or later (has API token counting).

    Claude 3 models: claude-3-opus, claude-3-sonnet, claude-3-haiku
    Claude 3.5 models: claude-3-5-sonnet, claude-3-5-haiku
    Claude 4 models: claude-sonnet-4, claude-opus-4
    """
    model_lower = model.lower()
    return any(
        x in model_lower
        for x in ["claude-3", "claude-4", "claude-sonnet-4", "claude-opus-4"]
    )


def is_openai_model(model: str) -> bool:
    """Check if model is from OpenAI."""
    model_lower = model.lower()
    return any(x in model_lower for x in ["gpt", "openai", "o1", "o3"])


def get_tokenizer(model: str | None = None, offline: bool = False) -> Tokenizer:
    """
    Select appropriate tokenizer based on model name.

    Priority:
    1. Anthropic API (for claude-3+ models, if not offline)
    2. Tiktoken with model-specific encoding (for OpenAI models)
    3. Tiktoken with cl100k_base fallback

    Args:
        model: Model name (e.g., "anthropic/claude-sonnet-4-20250514", "gpt-4")
        offline: If True, skip API-based tokenizers

    Returns:
        Appropriate Tokenizer instance
    """
    if model:
        if not offline and is_anthropic_model(model) and is_claude_3_or_later(model):
            return AnthropicAPITokenizer(model)

        if is_openai_model(model):
            # Extract model name without provider prefix
            model_name = model.split("/")[-1] if "/" in model else model
            return TiktokenTokenizer(model=model_name)

    # Fallback
    return TiktokenTokenizer(encoding="cl100k_base")


def estimate_image_tokens(image: dict[str, Any], model: str | None = None) -> int:
    """
    Estimate tokens for an image based on provider rules.

    Args:
        image: Image dict with url and optional detail
        model: Model name to determine estimation strategy

    Returns:
        Estimated token count for the image
    """
    if model and is_openai_model(model):
        # OpenAI uses detail level
        detail = image.get("image_url", {}).get("detail", "auto")
        if detail == "low":
            return 85
        return 1000  # Conservative high detail estimate

    elif model and is_anthropic_model(model):
        # Anthropic: ~1,600 tokens per image (varies by size)
        return 1600

    # Fallback
    return 1000

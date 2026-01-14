"""
Token categorization logic for CTX analysis.

Categorizes tokens by message role:
- system: System prompt messages
- user: User messages
- assistant: Assistant responses (text content only)
- tools: Tool definitions, tool calls, and tool responses
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel

from .tokenizers import Tokenizer


class TokenCategory(str, Enum):
    """Token categories for analysis."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOLS = "tools"


class CategoryBreakdown(BaseModel):
    """Detailed breakdown within a category."""

    total: int = 0
    # For tools category
    tool_definitions: int = 0
    tool_calls: int = 0
    tool_responses: int = 0
    # For user category
    text_content: int = 0
    images: int = 0


class TokenCategories(BaseModel):
    """Token counts by category."""

    system: int = 0
    user: int = 0
    assistant: int = 0
    tools: int = 0

    # Detailed breakdowns
    tools_breakdown: CategoryBreakdown = CategoryBreakdown()
    user_breakdown: CategoryBreakdown = CategoryBreakdown()

    def total(self) -> int:
        """Get total tokens across all categories."""
        return self.system + self.user + self.assistant + self.tools

    def as_dict(self) -> dict[str, int]:
        """Return simple dict of category totals."""
        return {
            "system": self.system,
            "user": self.user,
            "assistant": self.assistant,
            "tools": self.tools,
        }


def categorize_message(
    message: dict[str, Any], tokenizer: Tokenizer
) -> tuple[TokenCategory, int, CategoryBreakdown | None]:
    """
    Categorize a message and count its tokens.

    Args:
        message: Message dict with role and content
        tokenizer: Tokenizer to use for counting

    Returns:
        Tuple of (category, token_count, optional_breakdown)
    """
    role = message.get("role", "")
    tokens = tokenizer.count_message_tokens(message)
    breakdown = None

    if role == "system":
        return TokenCategory.SYSTEM, tokens, None

    elif role == "user":
        # Check for image content
        breakdown = CategoryBreakdown(total=tokens)
        content = message.get("content", "")

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "image_url":
                        # Estimate image tokens
                        breakdown.images += tokenizer._estimate_image_tokens(item) if hasattr(tokenizer, '_estimate_image_tokens') else 1000
                    elif item.get("type") == "text":
                        breakdown.text_content += tokenizer.count_tokens(item.get("text", ""))
        else:
            breakdown.text_content = tokens

        return TokenCategory.USER, tokens, breakdown

    elif role == "assistant":
        # Check if this is primarily a tool call message
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            # Count tool call tokens
            breakdown = CategoryBreakdown(total=tokens)
            tool_call_tokens = 0
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    func = tool_call.get("function", {})
                    tool_call_tokens += tokenizer.count_tokens(func.get("name", ""))
                    tool_call_tokens += tokenizer.count_tokens(func.get("arguments", ""))
                    tool_call_tokens += 10  # Overhead
            breakdown.tool_calls = tool_call_tokens
            return TokenCategory.TOOLS, tokens, breakdown

        return TokenCategory.ASSISTANT, tokens, None

    elif role == "tool":
        # Tool response
        breakdown = CategoryBreakdown(total=tokens, tool_responses=tokens)
        return TokenCategory.TOOLS, tokens, breakdown

    # Unknown role - categorize as user
    return TokenCategory.USER, tokens, None


def categorize_tools(
    tools: list[dict[str, Any]], tokenizer: Tokenizer
) -> tuple[int, CategoryBreakdown]:
    """
    Count tokens in tool definitions.

    Args:
        tools: List of tool definition dicts
        tokenizer: Tokenizer to use for counting

    Returns:
        Tuple of (total_tokens, breakdown)
    """
    import json

    total_tokens = 0
    breakdown = CategoryBreakdown()

    for tool in tools:
        # Serialize the tool definition and count tokens
        tool_str = json.dumps(tool)
        tokens = tokenizer.count_tokens(tool_str)
        total_tokens += tokens
        breakdown.tool_definitions += tokens

    breakdown.total = total_tokens
    return total_tokens, breakdown


def categorize_messages(
    messages: list[dict[str, Any]],
    tokenizer: Tokenizer,
    tools: list[dict[str, Any]] | None = None,
) -> TokenCategories:
    """
    Categorize all messages and count tokens by category.

    Args:
        messages: List of message dicts
        tokenizer: Tokenizer to use for counting
        tools: Optional list of tool definitions

    Returns:
        TokenCategories with counts per category
    """
    categories = TokenCategories()
    tools_breakdown = CategoryBreakdown()
    user_breakdown = CategoryBreakdown()

    # Count tool definitions if provided
    if tools:
        tool_tokens, tool_bd = categorize_tools(tools, tokenizer)
        categories.tools += tool_tokens
        tools_breakdown.tool_definitions = tool_bd.tool_definitions
        tools_breakdown.total += tool_tokens

    # Categorize each message
    for message in messages:
        category, tokens, breakdown = categorize_message(message, tokenizer)

        if category == TokenCategory.SYSTEM:
            categories.system += tokens
        elif category == TokenCategory.USER:
            categories.user += tokens
            if breakdown:
                user_breakdown.text_content += breakdown.text_content
                user_breakdown.images += breakdown.images
        elif category == TokenCategory.ASSISTANT:
            categories.assistant += tokens
        elif category == TokenCategory.TOOLS:
            categories.tools += tokens
            if breakdown:
                tools_breakdown.tool_calls += breakdown.tool_calls
                tools_breakdown.tool_responses += breakdown.tool_responses
                tools_breakdown.total += tokens

    # Update totals in breakdowns
    user_breakdown.total = categories.user
    tools_breakdown.total = categories.tools

    categories.tools_breakdown = tools_breakdown
    categories.user_breakdown = user_breakdown

    return categories

"""
TinyLoop CTX - Context Analyzer

A diagnostic tool for analyzing LLM conversation token usage and detecting
when conversations enter the "dumb zone" - a region of the context window
where model performance may degrade.

Usage:
    # CLI
    tinyloop ctx conversation.json
    tinyloop ctx -s conversation.json  # simple output
    cat conversation.json | tinyloop ctx

    # Programmatic API
    from tinyloop.ctx import CTXAnalyzer, CTXResult, CTXStatus

    analyzer = CTXAnalyzer(model="anthropic/claude-sonnet-4-20250514")
    result = analyzer.analyze(messages)
    status = analyzer.status(messages)

    # LLM Integration
    from tinyloop import LLM
    from tinyloop.ctx import CTXMiddleware

    ctx = CTXMiddleware(threshold=0.4, action="warn")
    llm = LLM(model="anthropic/claude-sonnet-4-20250514")
    llm(prompt="Hello!")
    status = ctx.check(llm)
"""

from .analyzer import (
    CTXAnalyzer,
    CTXResult,
    CTXStatus,
    RoundResult,
    analyze_conversation,
    get_status,
)
from .categories import (
    CategoryBreakdown,
    TokenCategories,
    TokenCategory,
    categorize_message,
    categorize_messages,
)
from .middleware import (
    CTXMiddleware,
    CTXThresholdExceeded,
    create_ctx_hook,
)
from .tokenizers import (
    AnthropicAPITokenizer,
    TiktokenTokenizer,
    Tokenizer,
    estimate_image_tokens,
    get_tokenizer,
    is_anthropic_model,
    is_claude_3_or_later,
    is_openai_model,
)

__all__ = [
    # Analyzer
    "CTXAnalyzer",
    "CTXResult",
    "CTXStatus",
    "RoundResult",
    "analyze_conversation",
    "get_status",
    # Categories
    "TokenCategory",
    "TokenCategories",
    "CategoryBreakdown",
    "categorize_message",
    "categorize_messages",
    # Middleware
    "CTXMiddleware",
    "CTXThresholdExceeded",
    "create_ctx_hook",
    # Tokenizers
    "Tokenizer",
    "TiktokenTokenizer",
    "AnthropicAPITokenizer",
    "get_tokenizer",
    "is_anthropic_model",
    "is_claude_3_or_later",
    "is_openai_model",
    "estimate_image_tokens",
]

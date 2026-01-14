"""
Core analysis logic for CTX.

Provides CTXAnalyzer class for analyzing conversation token usage
and detecting when conversations enter the "dumb zone".
"""

from typing import Any

from pydantic import BaseModel

from .categories import TokenCategories, categorize_message, categorize_messages
from .tokenizers import Tokenizer, get_tokenizer


class RoundResult(BaseModel):
    """Token information for a single round (message)."""

    round_number: int
    role: str
    tokens: int
    cumulative_tokens: int
    percentage_used: float
    is_in_dumb_zone: bool


class CTXResult(BaseModel):
    """Full analysis result."""

    model: str | None
    context_window: int
    threshold: float
    threshold_tokens: int
    total_tokens: int
    percentage_used: float
    is_in_dumb_zone: bool
    dumb_zone_round: int | None  # First round that entered dumb zone
    categories: dict[str, int]  # Tokens per category
    categories_detail: TokenCategories  # Detailed breakdown
    rounds: list[RoundResult]
    tokenizer_used: str  # 'anthropic_api', 'tiktoken', 'fallback'


class CTXStatus(BaseModel):
    """Simple status check result."""

    total_tokens: int
    threshold_tokens: int
    context_window: int
    percentage_used: float
    is_in_dumb_zone: bool
    is_approaching: bool  # Within 10% of threshold
    tokens_until_dumb_zone: int  # Positive = safe, negative = over
    message: str  # Pre-formatted status message


class CTXAnalyzer:
    """
    Analyzer for conversation token usage and dumb zone detection.

    The "dumb zone" refers to the portion of the context window where
    model performance may degrade (default: after 40% utilization).
    """

    DEFAULT_CONTEXT_WINDOW = 168_000
    DEFAULT_THRESHOLD = 0.4

    def __init__(
        self,
        model: str | None = None,
        context_window: int | None = None,
        threshold: float | None = None,
        offline: bool = False,
    ):
        """
        Initialize the CTX analyzer.

        Args:
            model: Model name for tokenizer selection
            context_window: Context window size in tokens (default: 168,000)
            threshold: Dumb zone threshold as fraction (default: 0.4 = 40%)
            offline: If True, use offline tokenization only (no API calls)
        """
        self.model = model
        self.context_window = context_window or self.DEFAULT_CONTEXT_WINDOW
        self.threshold = threshold if threshold is not None else self.DEFAULT_THRESHOLD
        self.offline = offline

        # Calculate threshold in tokens
        self.threshold_tokens = int(self.context_window * self.threshold)

        # Initialize tokenizer
        self._tokenizer = get_tokenizer(model, offline)

    @property
    def tokenizer(self) -> Tokenizer:
        """Get the tokenizer instance."""
        return self._tokenizer

    def analyze(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> CTXResult:
        """
        Perform full analysis of a conversation.

        Args:
            messages: List of message dicts (OpenAI format)
            tools: Optional list of tool definitions

        Returns:
            CTXResult with full analysis
        """
        # Get category breakdown
        categories = categorize_messages(messages, self._tokenizer, tools)

        # Build round-by-round analysis
        rounds: list[RoundResult] = []
        cumulative = 0
        dumb_zone_round = None

        # Add tool definition tokens to initial cumulative if present
        if tools:
            import json

            for tool in tools:
                cumulative += self._tokenizer.count_tokens(json.dumps(tool))

        # Analyze each message as a round
        for i, message in enumerate(messages):
            category, tokens, _ = categorize_message(message, self._tokenizer)
            cumulative += tokens
            percentage = cumulative / self.context_window
            in_dumb_zone = cumulative >= self.threshold_tokens

            if in_dumb_zone and dumb_zone_round is None:
                dumb_zone_round = i + 1

            rounds.append(
                RoundResult(
                    round_number=i + 1,
                    role=message.get("role", "unknown"),
                    tokens=tokens,
                    cumulative_tokens=cumulative,
                    percentage_used=percentage,
                    is_in_dumb_zone=in_dumb_zone,
                )
            )

        total_tokens = categories.total()
        percentage_used = total_tokens / self.context_window

        return CTXResult(
            model=self.model,
            context_window=self.context_window,
            threshold=self.threshold,
            threshold_tokens=self.threshold_tokens,
            total_tokens=total_tokens,
            percentage_used=percentage_used,
            is_in_dumb_zone=total_tokens >= self.threshold_tokens,
            dumb_zone_round=dumb_zone_round,
            categories=categories.as_dict(),
            categories_detail=categories,
            rounds=rounds,
            tokenizer_used=self._tokenizer.name,
        )

    def status(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> CTXStatus:
        """
        Get simple status check for a conversation.

        This is faster than full analysis for quick checks.

        Args:
            messages: List of message dicts (OpenAI format)
            tools: Optional list of tool definitions

        Returns:
            CTXStatus with simple status information
        """
        # Count total tokens
        total_tokens = self._tokenizer.count_messages_tokens(messages)

        # Add tool definition tokens if present
        if tools:
            import json

            for tool in tools:
                total_tokens += self._tokenizer.count_tokens(json.dumps(tool))

        percentage_used = total_tokens / self.context_window
        tokens_until = self.threshold_tokens - total_tokens
        is_in_dumb_zone = total_tokens >= self.threshold_tokens

        # Within 10% of threshold = approaching
        approaching_threshold = self.threshold_tokens * 0.9
        is_approaching = total_tokens >= approaching_threshold and not is_in_dumb_zone

        # Format status message
        message = self._format_status_message(
            total_tokens=total_tokens,
            threshold_tokens=self.threshold_tokens,
            percentage_used=percentage_used,
            tokens_until=tokens_until,
            is_in_dumb_zone=is_in_dumb_zone,
            is_approaching=is_approaching,
        )

        return CTXStatus(
            total_tokens=total_tokens,
            threshold_tokens=self.threshold_tokens,
            context_window=self.context_window,
            percentage_used=percentage_used,
            is_in_dumb_zone=is_in_dumb_zone,
            is_approaching=is_approaching,
            tokens_until_dumb_zone=tokens_until,
            message=message,
        )

    def _format_status_message(
        self,
        total_tokens: int,
        threshold_tokens: int,
        percentage_used: float,
        tokens_until: int,
        is_in_dumb_zone: bool,
        is_approaching: bool,
    ) -> str:
        """Format a human-readable status message."""
        pct_str = f"{percentage_used * 100:.1f}%"
        tokens_str = f"{total_tokens:,} / {threshold_tokens:,} tokens"

        if is_in_dumb_zone:
            over = abs(tokens_until)
            return f"! {tokens_str} ({pct_str}) -- IN DUMB ZONE ({over:,} tokens over threshold)"
        elif is_approaching:
            return f"* {tokens_str} ({pct_str}) -- {tokens_until:,} tokens until dumb zone (warning: approaching)"
        else:
            return f"  {tokens_str} ({pct_str}) -- {tokens_until:,} tokens until dumb zone"


def analyze_conversation(
    messages: list[dict[str, Any]],
    model: str | None = None,
    context_window: int | None = None,
    threshold: float | None = None,
    tools: list[dict[str, Any]] | None = None,
    offline: bool = False,
) -> CTXResult:
    """
    Convenience function to analyze a conversation.

    Args:
        messages: List of message dicts (OpenAI format)
        model: Model name for tokenizer selection
        context_window: Context window size in tokens
        threshold: Dumb zone threshold as fraction
        tools: Optional list of tool definitions
        offline: If True, use offline tokenization only

    Returns:
        CTXResult with full analysis
    """
    analyzer = CTXAnalyzer(
        model=model,
        context_window=context_window,
        threshold=threshold,
        offline=offline,
    )
    return analyzer.analyze(messages, tools)


def get_status(
    messages: list[dict[str, Any]],
    model: str | None = None,
    context_window: int | None = None,
    threshold: float | None = None,
    tools: list[dict[str, Any]] | None = None,
    offline: bool = False,
) -> CTXStatus:
    """
    Convenience function to get conversation status.

    Args:
        messages: List of message dicts (OpenAI format)
        model: Model name for tokenizer selection
        context_window: Context window size in tokens
        threshold: Dumb zone threshold as fraction
        tools: Optional list of tool definitions
        offline: If True, use offline tokenization only

    Returns:
        CTXStatus with simple status information
    """
    analyzer = CTXAnalyzer(
        model=model,
        context_window=context_window,
        threshold=threshold,
        offline=offline,
    )
    return analyzer.status(messages, tools)

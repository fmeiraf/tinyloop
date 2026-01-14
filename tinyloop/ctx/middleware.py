"""
LLM class integration middleware for CTX.

Provides CTXMiddleware for monitoring context usage during LLM conversations
and triggering actions when the dumb zone is reached.
"""

import sys
from typing import Any, Callable

from .analyzer import CTXAnalyzer, CTXResult, CTXStatus


class CTXThresholdExceeded(Exception):
    """Raised when dumb zone threshold is exceeded (action='raise')."""

    def __init__(self, total_tokens: int, threshold_tokens: int, percentage_used: float):
        self.total_tokens = total_tokens
        self.threshold_tokens = threshold_tokens
        self.percentage_used = percentage_used
        super().__init__(
            f"Dumb zone threshold exceeded: {percentage_used:.1%} "
            f"({total_tokens:,} / {threshold_tokens:,} tokens)"
        )


class CTXMiddleware:
    """
    Middleware for monitoring LLM context usage.

    Provides hooks for checking context status during conversations
    and triggering actions when the dumb zone is reached.

    Usage:
        from tinyloop import LLM
        from tinyloop.ctx import CTXMiddleware

        # Create middleware
        ctx = CTXMiddleware(
            context_window=168000,
            threshold=0.4,
            action="warn"  # or "raise"
        )

        # Use with LLM instance
        llm = LLM(model="anthropic/claude-sonnet-4-20250514")
        llm(prompt="Hello!")

        # Check status at any point
        status = ctx.check(llm)
        print(status.message)

        # Or get full analysis
        result = ctx.analyze(llm)
    """

    def __init__(
        self,
        context_window: int = 168_000,
        threshold: float = 0.4,
        action: str = "warn",
        model: str | None = None,
        offline: bool = False,
        on_warning: Callable[[CTXStatus], None] | None = None,
    ):
        """
        Initialize CTX middleware.

        Args:
            context_window: Context window size in tokens
            threshold: Dumb zone threshold as fraction (0.0-1.0)
            action: Action when threshold exceeded - "warn" or "raise"
            model: Model name for tokenizer selection (inferred from LLM if not provided)
            offline: Use offline tokenization only
            on_warning: Optional callback for custom warning handling
        """
        self.context_window = context_window
        self.threshold = threshold
        self.action = action
        self.model = model
        self.offline = offline
        self.on_warning = on_warning

        # Internal state
        self._analyzer: CTXAnalyzer | None = None
        self._last_status: CTXStatus | None = None
        self._last_result: CTXResult | None = None

    def _get_analyzer(self, llm: Any = None) -> CTXAnalyzer:
        """Get or create the analyzer instance."""
        model = self.model
        if model is None and llm is not None:
            model = getattr(llm, "model", None)

        if self._analyzer is None or (model and self._analyzer.model != model):
            self._analyzer = CTXAnalyzer(
                model=model,
                context_window=self.context_window,
                threshold=self.threshold,
                offline=self.offline,
            )
        return self._analyzer

    def check(
        self,
        llm: Any = None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> CTXStatus:
        """
        Check current context status.

        Args:
            llm: LLM instance (extracts history automatically)
            messages: Alternatively, provide messages directly
            tools: Optional tool definitions

        Returns:
            CTXStatus with current status

        Raises:
            CTXThresholdExceeded: If action="raise" and threshold exceeded
        """
        if messages is None:
            if llm is None:
                raise ValueError("Either llm or messages must be provided")
            messages = llm.get_history()

        analyzer = self._get_analyzer(llm)
        status = analyzer.status(messages, tools)
        self._last_status = status

        # Handle threshold exceeded
        if status.is_in_dumb_zone:
            self._handle_threshold_exceeded(status)

        return status

    def analyze(
        self,
        llm: Any = None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> CTXResult:
        """
        Perform full analysis of current context.

        Args:
            llm: LLM instance (extracts history automatically)
            messages: Alternatively, provide messages directly
            tools: Optional tool definitions

        Returns:
            CTXResult with full analysis

        Raises:
            CTXThresholdExceeded: If action="raise" and threshold exceeded
        """
        if messages is None:
            if llm is None:
                raise ValueError("Either llm or messages must be provided")
            messages = llm.get_history()

        analyzer = self._get_analyzer(llm)
        result = analyzer.analyze(messages, tools)
        self._last_result = result

        # Handle threshold exceeded
        if result.is_in_dumb_zone:
            status = CTXStatus(
                total_tokens=result.total_tokens,
                threshold_tokens=result.threshold_tokens,
                context_window=result.context_window,
                percentage_used=result.percentage_used,
                is_in_dumb_zone=result.is_in_dumb_zone,
                is_approaching=False,
                tokens_until_dumb_zone=result.threshold_tokens - result.total_tokens,
                message="",
            )
            self._handle_threshold_exceeded(status)

        return result

    def _handle_threshold_exceeded(self, status: CTXStatus) -> None:
        """Handle threshold exceeded based on configured action."""
        if self.action == "raise":
            raise CTXThresholdExceeded(
                total_tokens=status.total_tokens,
                threshold_tokens=status.threshold_tokens,
                percentage_used=status.percentage_used,
            )
        elif self.action == "warn":
            if self.on_warning:
                self.on_warning(status)
            else:
                self._default_warning(status)

    def _default_warning(self, status: CTXStatus) -> None:
        """Default warning handler - prints to stderr."""
        print(
            f"\n[CTX WARNING] Dumb zone reached: {status.percentage_used:.1%} "
            f"({status.total_tokens:,} / {status.threshold_tokens:,} tokens)\n"
            f"Consider summarizing or pruning context before continuing.\n",
            file=sys.stderr,
        )

    def get_status(self) -> CTXStatus | None:
        """Get the last checked status."""
        return self._last_status

    def get_analysis(self) -> CTXResult | None:
        """Get the last full analysis."""
        return self._last_result

    # Aliases for compatibility with spec
    get_last_status = get_status
    get_last_analysis = get_analysis


def create_ctx_hook(
    context_window: int = 168_000,
    threshold: float = 0.4,
    action: str = "warn",
    offline: bool = False,
) -> Callable[[Any], CTXStatus]:
    """
    Create a simple hook function for checking LLM context status.

    This is a convenience function for simple use cases where you
    just want to check status after each LLM call.

    Usage:
        from tinyloop import LLM
        from tinyloop.ctx import create_ctx_hook

        llm = LLM(model="anthropic/claude-sonnet-4-20250514")
        check_ctx = create_ctx_hook(threshold=0.4)

        # After each call
        response = llm(prompt="Hello!")
        status = check_ctx(llm)
        if status.is_in_dumb_zone:
            print("Warning: In dumb zone!")

    Args:
        context_window: Context window size in tokens
        threshold: Dumb zone threshold as fraction
        action: Action when threshold exceeded - "warn" or "raise"
        offline: Use offline tokenization only

    Returns:
        Callable that takes an LLM instance and returns CTXStatus
    """
    middleware = CTXMiddleware(
        context_window=context_window,
        threshold=threshold,
        action=action,
        offline=offline,
    )
    return middleware.check

"""
TinyLoop - A super lightweight library for LLM-based applications
"""

from importlib import import_module
from typing import Any

# Export main classes (lazily imported to avoid import-time side effects like MLflow autolog)
__all__ = ["LLM", "Generate", "ToolLoop"]

# Version info
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    """
    Lazy attribute access for top-level imports.

    This prevents import-time side effects (e.g., MLflow autolog setup in ToolLoop)
    when users only need `LLM`.
    """

    if name == "LLM":
        return import_module("tinyloop.inference.litellm").LLM
    if name == "Generate":
        return import_module("tinyloop.modules.generate").Generate
    if name == "ToolLoop":
        return import_module("tinyloop.modules.tool_loop").ToolLoop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)

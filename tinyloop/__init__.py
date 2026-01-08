"""
TinyLoop - A super lightweight library for LLM-based applications
"""

from importlib import import_module
from typing import Any

# Export main classes
__all__ = ["LLM", "Generate"]

# Version info
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    """Lazy attribute access for top-level imports."""

    if name == "LLM":
        return import_module("tinyloop.inference.litellm").LLM
    if name == "Generate":
        return import_module("tinyloop.modules.generate").Generate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)

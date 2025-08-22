"""Functionality modules for tinyloop."""

from .function_calling import function_to_tool_json, tool

__all__ = [
    "tool",
    "function_to_tool_json",
]

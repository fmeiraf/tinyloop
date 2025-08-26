"""
TinyLoop - A super lightweight library for LLM-based applications
"""

# Import main classes for easier access
from tinyloop.inference.litellm import LLM
from tinyloop.modules.generate import Generate
from tinyloop.modules.tool_loop import ToolLoop

# Export main classes
__all__ = ["LLM", "Generate", "ToolLoop"]

# Version info
__version__ = "0.1.0"

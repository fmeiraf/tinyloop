"""Clean function calling module for converting Python functions to JSON tool definitions."""

import inspect
import re
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import mlflow

from tinyloop.types import ToolCallResponse
from tinyloop.utils.observability import set_trace_custom

mlflow.config.enable_async_logging(True)


class Tool:
    """
    A tool wrapper that converts a Python function to JSON tool definition.

    Args:
        func: The function to convert to a tool
        hidden_params: List of parameter names to omit from the JSON signature

    Example:
        def get_weather(location: str, unit: str, context: dict):
            '''Get weather for a location'''
            return f"Weather in {location}"

        weather_tool = Tool(get_weather, hidden_params=['context'])
        tool_json = weather_tool.definition
    """

    def __init__(
        self,
        func: Callable,
        hidden_params: Optional[List[str]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self.func = func
        self.hidden_params = hidden_params or []
        self.name = name or func.__name__
        self.description = description

        self.definition = function_to_tool_json(
            func, self.name, self.description, self.hidden_params
        )

    @set_trace_custom(
        mlflow.entities.SpanType.TOOL, lambda self, func: f"{self.name}.{func.__name__}"
    )
    def __call__(self, *args, **kwargs) -> ToolCallResponse:
        """Allow the tool to be called like the original function."""
        tool_result = self.func(*args, **kwargs)
        return tool_result

    @set_trace_custom(
        mlflow.entities.SpanType.TOOL, lambda self, func: f"{self.name}.{func.__name__}"
    )
    async def acall(self, *args, **kwargs) -> ToolCallResponse:
        """Allow the tool to be called like the original function."""
        if inspect.iscoroutinefunction(self.func):
            tool_result = await self.func(*args, **kwargs)
        else:
            tool_result = self.func(*args, **kwargs)
        return tool_result


def function_to_tool_json(
    func: Callable,
    name: Optional[str],
    description: Optional[str] = None,
    hidden_params: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Convert a Python function to OpenAI function calling JSON format.

    Args:
        func: The function to convert
        hidden_params: List of parameter names to omit from the JSON signature

    Returns:
        Dictionary in OpenAI function calling format
    """
    hidden_params = hidden_params or []

    # Get function signature and docstring
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""

    # Parse docstring for description and parameter docs
    doc_description, param_docs = _parse_docstring(doc)
    description = description or doc_description

    # Get type hints
    type_hints = get_type_hints(func)

    # Build parameters
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name in hidden_params:
            continue

        # Get parameter type
        param_type = type_hints.get(param_name, param.annotation)
        json_type, enum_values = _python_type_to_json_schema(param_type)

        # Build property definition
        prop_def = {"type": json_type}

        # Add description from docstring and extract enum values
        if param_name in param_docs:
            desc = param_docs[param_name]

            # Check for enum values in description (e.g., {'celsius', 'fahrenheit'})
            enum_match = re.search(r"\{([^}]+)\}", desc)
            if enum_match:
                enum_str = enum_match.group(1)
                # Parse comma-separated values, handling quotes
                enum_items = [item.strip().strip("'\"") for item in enum_str.split(",")]
                if enum_items and all(item for item in enum_items):
                    enum_values = enum_items
                    # Remove the enum part from the description
                    desc = re.sub(r"\s*\{[^}]+\}\s*", " ", desc).strip()

            prop_def["description"] = desc

        # Add enum if present
        if enum_values:
            prop_def["enum"] = enum_values

        properties[param_name] = prop_def

        # Add to required if no default value
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": name or func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _parse_docstring(docstring: str) -> tuple[str, Dict[str, str]]:
    """
    Parse function docstring to extract description and parameter documentation.

    Supports both Google-style and NumPy-style docstrings.
    """
    if not docstring:
        return "", {}

    lines = docstring.strip().split("\n")
    description_lines = []
    param_docs = {}

    in_params_section = False
    current_param = None
    description_done = False

    for i, line in enumerate(lines):
        line = line.strip()

        # Check for section headers (Parameters, Returns, etc.)
        if line.lower() in ["parameters", "parameters:", "args:", "arguments:"] or (
            line.lower().startswith("parameters")
            and i + 1 < len(lines)
            and set(lines[i + 1].strip()) <= {"-", "="}
        ):
            in_params_section = True
            description_done = True
            continue

        # Check for other sections that end parameters
        if line.lower().startswith(
            ("returns", "yields", "raises", "examples", "notes", "see also")
        ):
            in_params_section = False
            description_done = True
            continue

        # Skip separator lines
        if set(line) <= {"-", "=", " "}:
            continue

        if in_params_section:
            # Check for parameter definition
            param_match = re.match(r"(\w+)\s*:\s*(.+)", line)
            if param_match:
                current_param = param_match.group(1)
                param_type_desc = param_match.group(2)
                # For NumPy style, the whole line after the colon is the description
                # Extract description (everything after the basic type, but preserve enum info)
                desc_match = re.search(r"^[^{]*(\{[^}]*\})?\s*(.*)$", param_type_desc)
                if desc_match:
                    enum_part = desc_match.group(1) or ""
                    desc_part = desc_match.group(2) or ""
                    full_desc = (enum_part + " " + desc_part).strip()
                    param_docs[current_param] = full_desc
                else:
                    param_docs[current_param] = param_type_desc.strip()
            elif current_param and line:
                # Continuation of parameter description
                if current_param in param_docs:
                    param_docs[current_param] += " " + line
                else:
                    param_docs[current_param] = line
        elif not description_done:
            # Build description from lines before any section
            if line:
                description_lines.append(line)

    description = " ".join(description_lines).strip()

    # Clean up parameter descriptions
    for param in param_docs:
        param_docs[param] = param_docs[param].strip()

    return description, param_docs


def _python_type_to_json_schema(python_type: Any) -> tuple[str, Optional[List[str]]]:
    """
    Convert Python type annotation to JSON Schema type and enum values.

    Returns:
        Tuple of (json_type, enum_values)
    """
    # Handle None/empty annotation
    if python_type == inspect.Parameter.empty or python_type is None:
        return "string", None

    # Handle Union types (including Optional)
    origin = get_origin(python_type)
    if origin is Union:
        args = get_args(python_type)
        # Filter out NoneType for Optional handling
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return _python_type_to_json_schema(non_none_args[0])
        # For multiple non-None types, default to string
        return "string", None

    # Handle basic types
    if python_type is str or python_type == "str":
        return "string", None
    elif python_type is int or python_type == "int":
        return "integer", None
    elif python_type is float or python_type == "float":
        return "number", None
    elif python_type is bool or python_type == "bool":
        return "boolean", None
    elif python_type is list or origin is list:
        return "array", None
    elif python_type is dict or origin is dict:
        return "object", None

    # Handle string type hints with enum-like information in docstring
    # This is a simple heuristic - could be enhanced
    return "string", None

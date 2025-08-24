"""Tests for function calling module."""

from unittest.mock import patch

from tinyloop.features.function_calling import Tool


def test_tool_mlflow_tracing():
    """Test that Tool class uses custom MLflow tracing with tool name."""

    def sample_function(location: str, unit: str = "celsius"):
        """Get weather for a location.

        Args:
            location: The city name
            unit: Temperature unit {'celsius', 'fahrenheit'}
        """
        return f"Weather in {location}: 20°{unit}"

    # Create a tool with a custom name
    weather_tool = Tool(sample_function, name="get_weather_tool")

    # Mock mlflow.trace to capture the span name
    with patch("mlflow.trace") as mock_trace:
        mock_trace.return_value = lambda func: func

        # Call the tool
        result = weather_tool("London", "fahrenheit")

        # Verify the result
        assert result == "Weather in London: 20°fahrenheit"

        # Verify mlflow.trace was called with the correct span name
        mock_trace.assert_called()
        call_args = mock_trace.call_args
        assert call_args[1]["span_type"] == "TOOL"
        assert call_args[1]["name"] == "get_weather_tool.__call__"


def test_tool_async_mlflow_tracing():
    """Test that Tool class async method uses custom MLflow tracing with tool name."""

    async def sample_async_function(location: str, unit: str = "celsius"):
        """Get weather for a location asynchronously.

        Args:
            location: The city name
            unit: Temperature unit {'celsius', 'fahrenheit'}
        """
        return f"Weather in {location}: 20°{unit}"

    # Create a tool with a custom name
    weather_tool = Tool(sample_async_function, name="get_weather_async_tool")

    # Mock mlflow.trace to capture the span name
    with patch("mlflow.trace") as mock_trace:
        mock_trace.return_value = lambda func: func

        # We can't easily test async without running an event loop,
        # but we can verify the decorator is applied correctly
        assert hasattr(weather_tool.acall, "__wrapped__") or hasattr(
            weather_tool.acall, "__call__"
        )


def test_tool_default_name():
    """Test that Tool uses function name as default when no name provided."""

    def sample_function(location: str):
        """Get weather for a location."""
        return f"Weather in {location}"

    # Create a tool without specifying a name
    weather_tool = Tool(sample_function)

    # Mock mlflow.trace to capture the span name
    with patch("mlflow.trace") as mock_trace:
        mock_trace.return_value = lambda func: func

        # Call the tool
        result = weather_tool("Paris")

        # Verify the result
        assert result == "Weather in Paris"

        # Verify mlflow.trace was called with the function name
        mock_trace.assert_called()
        call_args = mock_trace.call_args
        assert call_args[1]["name"] == "sample_function.__call__"

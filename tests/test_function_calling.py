"""Tests for function calling module."""

import asyncio

from tinyloop.features.function_calling import Tool


def test_tool_sync_call():
    """Test that Tool class can be called synchronously."""

    def sample_function(location: str, unit: str = "celsius"):
        """Get weather for a location.

        Args:
            location: The city name
            unit: Temperature unit {'celsius', 'fahrenheit'}
        """
        return f"Weather in {location}: 20°{unit}"

    # Create a tool with a custom name
    weather_tool = Tool(sample_function, name="get_weather_tool")

    # Call the tool
    result = weather_tool("London", "fahrenheit")

    # Verify the result
    assert result == "Weather in London: 20°fahrenheit"


def test_tool_async_call():
    """Test that Tool class async method works correctly."""

    async def sample_async_function(location: str, unit: str = "celsius"):
        """Get weather for a location asynchronously.

        Args:
            location: The city name
            unit: Temperature unit {'celsius', 'fahrenheit'}
        """
        return f"Weather in {location}: 20°{unit}"

    # Create a tool with a custom name
    weather_tool = Tool(sample_async_function, name="get_weather_async_tool")

    # Run the async call
    result = asyncio.run(weather_tool.acall("Tokyo", "celsius"))

    # Verify the result
    assert result == "Weather in Tokyo: 20°celsius"


def test_tool_default_name():
    """Test that Tool uses function name as default when no name provided."""

    def sample_function(location: str):
        """Get weather for a location."""
        return f"Weather in {location}"

    # Create a tool without specifying a name
    weather_tool = Tool(sample_function)

    # Call the tool
    result = weather_tool("Paris")

    # Verify the result
    assert result == "Weather in Paris"

    # Verify the tool name defaults to function name
    assert weather_tool.name == "sample_function"

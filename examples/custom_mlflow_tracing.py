"""Example demonstrating custom MLflow tracing with Tool class."""

import mlflow

from tinyloop.features.function_calling import Tool


def get_weather(location: str, unit: str = "celsius"):
    """Get weather for a location.

    Args:
        location: The city name
        unit: Temperature unit {'celsius', 'fahrenheit'}
    """
    return f"Weather in {location}: 20°{unit}"


def get_stock_price(symbol: str, currency: str = "USD"):
    """Get stock price for a symbol.

    Args:
        symbol: The stock symbol
        currency: The currency {'USD', 'EUR', 'GBP'}
    """
    return f"Stock price for {symbol}: $150.00 {currency}"


def main():
    """Demonstrate custom MLflow tracing with different tool names."""

    # Create tools with custom names
    weather_tool = Tool(get_weather, name="weather_service")
    stock_tool = Tool(get_stock_price, name="stock_service")

    # Start MLflow run
    with mlflow.start_run():
        # Call tools - these will create spans with custom names
        weather_result = weather_tool("London", "fahrenheit")
        stock_result = stock_tool("AAPL", "USD")

        print(f"Weather result: {weather_result}")
        print(f"Stock result: {stock_result}")

        # The MLflow spans will be named:
        # - "weather_service.__call__" for the weather tool
        # - "stock_service.__call__" for the stock tool
        # Instead of the default "Tool.__call__"


if __name__ == "__main__":
    main()

"""
TinyLoop - A super lightweight library for LLM-based applications
"""


def hello() -> str:
    return "Hello from tinyloop!"


def initialize_tracing():
    """Initialize MLflow tracing when needed."""
    try:
        from .observability.mlflow import initialize_mlflow

        return initialize_mlflow()
    except Exception as e:
        print(f"Warning: Failed to initialize tracing: {e}")
        return False


__all__ = ["hello", "initialize_tracing"]

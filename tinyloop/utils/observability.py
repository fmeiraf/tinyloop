import inspect
import os
from typing import Any, Callable, TypeVar, cast

import mlflow

F = TypeVar("F", bound=Callable[..., Any])


def _truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def observe(*args: Any, **kwargs: Any):
    """
    Safe wrapper around `langfuse.observe`.

    By default, this is a no-op to avoid noisy OTEL exporter errors (e.g. when no
    collector/Langfuse server is running). To enable Langfuse tracing, set:

        TINYLOOP_ENABLE_LANGFUSE=1
    """

    enabled = _truthy_env(os.getenv("TINYLOOP_ENABLE_LANGFUSE"))
    if not enabled:
        # Support both decorator styles:
        #  - @observe
        #  - @observe(name="...", as_type="...")
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return cast(F, args[0])

        def decorator(func: F) -> F:
            return func

        return decorator

    from langfuse import observe as _observe

    return _observe(*args, **kwargs)


# helper: set span name to "ClassName.method" using the function's qualname
def set_trace(span_type):
    def decorator(func):
        return mlflow.trace(span_type=span_type, name=func.__qualname__)(func)

    return decorator


# helper: set span name using a custom name function with better Langfuse compatibility
def set_trace_custom(span_type, name_func):
    """
    Custom MLflow trace decorator that uses a function to generate the span name.
    Properly handles both sync and async functions and is compatible with Langfuse observe.

    Args:
        span_type: The MLflow span type
        name_func: Function that takes the instance (self) and function, returns the span name

    Example:
        @set_trace_custom(mlflow.entities.SpanType.TOOL,
                           lambda self, func: f"{self.name}.{func.__name__}")
        def __call__(self, *args, **kwargs):
            return self.func(*args, **kwargs)
    """

    def decorator(func):
        # Check if the function is async
        if inspect.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                # Get the span name at call time
                if args and hasattr(
                    args[0], "__dict__"
                ):  # Check if first arg is likely 'self'
                    span_name = name_func(args[0], func)
                else:
                    span_name = name_func(None, func)

                # Apply both decorators separately to avoid conflicts
                # First apply MLflow tracing
                mlflow_traced = mlflow.trace(span_type=span_type, name=span_name)(func)
                # Then apply Langfuse observe
                langfuse_traced = observe(name=span_name)(mlflow_traced)

                return await langfuse_traced(*args, **kwargs)

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                # Get the span name at call time
                if args and hasattr(
                    args[0], "__dict__"
                ):  # Check if first arg is likely 'self'
                    span_name = name_func(args[0], func)
                else:
                    span_name = name_func(None, func)

                # Apply both decorators separately to avoid conflicts
                # First apply MLflow tracing
                mlflow_traced = mlflow.trace(span_type=span_type, name=span_name)(func)
                # Then apply Langfuse observe
                langfuse_traced = observe(name=span_name)(mlflow_traced)

                return langfuse_traced(*args, **kwargs)

            return sync_wrapper

    return decorator

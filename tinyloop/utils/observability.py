import inspect

import mlflow
from langfuse import observe


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

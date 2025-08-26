import inspect

import mlflow


# helper: set span name to "ClassName.method" using the function's qualname
def mlflow_trace(span_type):
    def decorator(func):
        return mlflow.trace(span_type=span_type, name=func.__qualname__)(func)

    return decorator


# helper: set span name using a custom name function
def mlflow_trace_custom(span_type, name_func):
    """
    Custom MLflow trace decorator that uses a function to generate the span name.
    Properly handles both sync and async functions.

    Args:
        span_type: The MLflow span type
        name_func: Function that takes the instance (self) and function, returns the span name

    Example:
        @mlflow_trace_custom(mlflow.entities.SpanType.TOOL,
                           lambda self, func: f"{self.name}.{func.__name__}")
        def __call__(self, *args, **kwargs):
            return self.func(*args, **kwargs)
    """

    def decorator(func):
        # Check if the function is async
        if inspect.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                # Create a new async function that captures the current span name
                async def traced_func(*inner_args, **inner_kwargs):
                    return await func(*inner_args, **inner_kwargs)

                # Get the span name at call time, passing self as first argument
                if args and hasattr(
                    args[0], "__dict__"
                ):  # Check if first arg is likely 'self'
                    span_name = name_func(args[0], func)
                else:
                    span_name = name_func(None, func)

                traced_func = mlflow.trace(span_type=span_type, name=span_name)(
                    traced_func
                )
                return await traced_func(*args, **kwargs)

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                # Create a new function that captures the current span name
                def traced_func(*inner_args, **inner_kwargs):
                    return func(*inner_args, **inner_kwargs)

                # Get the span name at call time, passing self as first argument
                if args and hasattr(
                    args[0], "__dict__"
                ):  # Check if first arg is likely 'self'
                    span_name = name_func(args[0], func)
                else:
                    span_name = name_func(None, func)

                traced_func = mlflow.trace(span_type=span_type, name=span_name)(
                    traced_func
                )
                return traced_func(*args, **kwargs)

            return sync_wrapper

    return decorator

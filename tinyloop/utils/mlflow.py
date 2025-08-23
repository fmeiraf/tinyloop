import mlflow


# helper: set span name to "ClassName.method" using the function's qualname
def mlflow_trace(span_type):
    def decorator(func):
        return mlflow.trace(span_type=span_type, name=func.__qualname__)(func)

    return decorator

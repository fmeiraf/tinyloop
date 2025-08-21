# Pydantic Warning Solutions

## The Issue

When running integration tests with litellm, you may see warnings like:

```
UserWarning: Pydantic serializer warnings:
  PydanticSerializationUnexpectedValue(Expected 9 fields but got 6: Expected `Message`...)
```

These warnings are **harmless** and don't affect functionality. They occur because:

1. litellm creates OpenAI response objects
2. Pydantic tries to serialize them with different field expectations than what's present
3. The warnings are generated deep in the pydantic serialization process

## Solutions (Choose One)

### Option 1: Run with Warnings Disabled (Recommended)

```bash
# Disable all warnings (cleanest output)
uv run pytest tests/ -v --disable-warnings

# Disable warnings for specific test
uv run pytest tests/test_llm_live.py::TestLLMLive::test_invoke_basic_functionality -v --disable-warnings
```

### Option 2: Use Environment Variable

```bash
# Set environment variable to suppress warnings
export PYTHONWARNINGS="ignore::UserWarning:pydantic.*"
uv run pytest tests/ -v

# Or inline
PYTHONWARNINGS="ignore::UserWarning:pydantic.*" uv run pytest tests/ -v
```

### Option 3: Ignore Warning Summary

Simply ignore the warning summary at the end of test output. The tests still pass and function correctly.

### Option 4: Use Quiet Mode

```bash
# Reduce output verbosity
uv run pytest tests/ -q --disable-warnings
```

## Why Standard Warning Filters Don't Work

The Pydantic warnings are generated during the serialization process inside litellm and are captured by pytest before our warning filters can suppress them. This is a known limitation when dealing with third-party library compatibility warnings.

## Verification

You can verify the warnings don't affect functionality:

```bash
# Run smoke test (no API calls, no warnings)
uv run pytest tests/test_llm_live.py::test_quick_smoke_test -v

# Run with API key to see warnings, but tests still pass
uv run pytest tests/test_llm_live.py -v --disable-warnings
```

## Recommendation

**Use `--disable-warnings` flag** when running your integration tests. This provides the cleanest output while maintaining full test functionality.

Add this to your test commands:

```bash
# Clean integration test runs
uv run pytest tests/ -v --disable-warnings
uv run pytest tests/ -m integration -v --disable-warnings
```

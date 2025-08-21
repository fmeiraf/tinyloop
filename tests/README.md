# TinyLoop Integration Tests

This directory contains live integration tests for the TinyLoop library that hit real APIs.

## Test Files

- `test_llm_live.py` - Core integration tests for LLM functionality
- `test_integration_litellm.py` - Comprehensive integration tests with multiple providers
- `conftest.py` - Test configuration and fixtures

## Running Tests

### Prerequisites

Set up API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"  # Optional
```

Or create a `.env` file in the project root:

```bash
# .env file
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

### Run All Integration Tests

```bash
# Run all integration tests
pytest tests/ -m integration -v

# Run specific test file
pytest tests/test_llm_live.py -v

# Run without API key requirements (smoke tests only)
pytest tests/test_llm_live.py::test_quick_smoke_test -v
```

### Run Individual Tests

```bash
# Test basic invoke functionality
pytest tests/test_llm_live.py::TestLLMLive::test_invoke_basic_functionality -v

# Test async functionality
pytest tests/test_llm_live.py::TestLLMLive::test_ainvoke_basic_functionality -v

# Test message history
pytest tests/test_llm_live.py::TestLLMLive::test_message_history_management -v
```

### Quick Test Run

```bash
# Run the test file directly for a quick check
python tests/test_llm_live.py
```

## Test Coverage

The integration tests cover:

1. **Basic Functionality**

   - `invoke()` method with real API calls
   - `ainvoke()` async method with real API calls
   - `__call__()` convenience method
   - `acall()` async convenience method

2. **Message History Management**

   - Getting message history
   - Setting message history
   - Adding messages to history
   - Using history in API calls

3. **Parameter Handling**

   - Temperature parameter
   - Model selection
   - Custom parameters via kwargs

4. **Multiple Providers**
   - OpenAI GPT models
   - Anthropic Claude models (if API key available)
   - Other litellm-supported providers

## Notes

- Tests require internet connection and valid API keys
- Some tests are skipped if API keys are not available
- Tests use low-cost models (gpt-3.5-turbo, claude-haiku) to minimize costs
- Temperature is set to 0.0 for deterministic testing where possible
- All tests include proper error handling and skip conditions
- **Pydantic warnings from litellm are automatically suppressed** for clean test output

## Troubleshooting

### Clean Test Output

If you see Pydantic serialization warnings like:

```
PydanticSerializationUnexpectedValue(Expected 9 fields but got 6...)
```

These are **harmless compatibility warnings** from litellm. For clean output, run tests with:

```bash
# Clean output (recommended)
uv run pytest tests/ -v --disable-warnings

# Or with environment variable
PYTHONWARNINGS="ignore::UserWarning:pydantic.*" uv run pytest tests/ -v
```

See `WARNINGS_SOLUTION.md` for detailed explanation and solutions.

### API Key Issues

If tests fail with authentication errors:

1. Verify your API key is correctly set
2. Check that the API key has sufficient permissions
3. Ensure you have credits/quota available

### Network Issues

If tests fail with connection errors:

1. Check your internet connection
2. Verify the API endpoints are accessible
3. Consider using a VPN if behind corporate firewall

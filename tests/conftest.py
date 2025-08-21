"""
Test configuration for tinyloop integration tests.
"""

import os
import warnings

import pytest

# Suppress Pydantic warnings globally for tests
# Most comprehensive approach - suppress ALL UserWarnings from pydantic.main
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.main")

# Additional specific patterns to catch the warning content
warnings.filterwarnings("ignore", message="Pydantic serializer warnings.*")
warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")
warnings.filterwarnings("ignore", message=".*Expected.*fields but got.*")
warnings.filterwarnings("ignore", message=".*serialized value may not be as expected.*")


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test that hits real APIs"
    )
    config.addinivalue_line(
        "markers", "requires_api_key: mark test as requiring API keys to run"
    )


@pytest.fixture
def openai_available():
    """Check if OpenAI API key is available."""
    return os.getenv("OPENAI_API_KEY") is not None


@pytest.fixture
def anthropic_available():
    """Check if Anthropic API key is available."""
    return os.getenv("ANTHROPIC_API_KEY") is not None


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Hello! Please respond with 'Hello, World!' and nothing else.",
        },
    ]


@pytest.fixture
def simple_question():
    """Simple question for testing."""
    return [{"role": "user", "content": "What is 2+2? Answer with just the number."}]

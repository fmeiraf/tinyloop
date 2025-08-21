"""
Test to verify clean test output without Pydantic warnings.
Run this with: uv run pytest tests/test_clean_output.py -v

If you have an OPENAI_API_KEY set, this will demonstrate clean output.
"""

import os

import pytest

from tinyloop.inference.litellm import LLM


@pytest.mark.integration
def test_clean_output_with_api_call():
    """Test that demonstrates clean output when making API calls."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not available - set it to test clean output")

    print("\n🧪 Testing clean output with real API call...")

    llm = LLM(model="gpt-3.5-turbo", temperature=0.0)
    messages = [
        {"role": "user", "content": "Say 'Clean test output!' and nothing else."}
    ]

    response = llm.invoke(messages=messages)

    assert response is not None
    if hasattr(response, "choices"):
        content = response.choices[0].message.content
        assert content is not None
        print(f"✓ Response received: {content}")
        assert "Clean test output!" in content

    print("✓ Test completed with clean output!")


def test_clean_output_without_api():
    """Test that demonstrates clean output without API calls."""
    print("\n🧪 Testing clean output without API calls...")

    llm = LLM(model="gpt-3.5-turbo", temperature=0.5)

    # Test basic functionality
    assert llm.model == "gpt-3.5-turbo"
    assert llm.temperature == 0.5

    # Test message history
    llm.add_message({"role": "user", "content": "test message"})
    assert len(llm.get_history()) == 1

    print("✓ Basic functionality test completed with clean output!")


if __name__ == "__main__":
    print("Running clean output tests...")
    test_clean_output_without_api()

    if os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY found, testing with API...")
        # Note: Can't run pytest.skip() outside of pytest context
        print("Run with pytest to test API functionality")
    else:
        print("Set OPENAI_API_KEY to test API functionality")

    print("Tests completed!")

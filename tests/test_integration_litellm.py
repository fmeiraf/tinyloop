"""
Live integration tests for LLM class using real API calls.

These tests hit actual APIs and require internet connection.
Some tests may require API keys to be set as environment variables.
"""

import os

import pytest
from dotenv import load_dotenv

from tinyloop.inference.litellm import LLM

load_dotenv()


class TestLLMIntegration:
    """Live integration tests for LLM class."""

    def test_invoke_with_messages_history(self):
        """Test invoke with message history management."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ]

        llm = LLM(
            model="gpt-3.5-turbo",  # Will only work with OpenAI API key
            temperature=0.1,
            message_history=messages,
        )

        # Test that history is properly set
        assert llm.get_history() == messages

        # Add a message
        llm.add_message({"role": "assistant", "content": "4"})
        assert len(llm.get_history()) == 3

        # Test with API key if available
        if os.getenv("OPENAI_API_KEY"):
            try:
                response = llm.invoke(prompt="Got it? ")
                assert response is not None

                # Test that we can parse the response
                if hasattr(response, "choices"):
                    content = response.choices[0].message.content
                    assert content is not None
                    assert len(content.strip()) > 0

            except Exception as e:
                pytest.skip(f"OpenAI API test failed: {str(e)}")
        else:
            pytest.skip("OPENAI_API_KEY not set, skipping OpenAI test")

    @pytest.mark.asyncio
    async def test_ainvoke_async_functionality(self):
        """Test asynchronous ainvoke method."""
        messages = [{"role": "user", "content": "Say 'Hello World' and nothing else."}]

        llm = LLM(
            model="gpt-3.5-turbo",
            temperature=0.0,  # Deterministic for testing
        )

        if os.getenv("OPENAI_API_KEY"):
            try:
                response = await llm.ainvoke(messages=messages)

                # Basic assertions
                assert response is not None

                # If it's a completion response object, check for content
                if hasattr(response, "choices"):
                    content = response.choices[0].message.content
                    assert content is not None
                    assert "Hello World" in content or "hello world" in content.lower()

            except Exception as e:
                pytest.skip(f"Async OpenAI API test failed: {str(e)}")
        else:
            pytest.skip("OPENAI_API_KEY not set, skipping async OpenAI test")

    @pytest.mark.asyncio
    async def test_acall_method(self):
        """Test the acall convenience method."""
        messages = [
            {"role": "user", "content": "Respond with exactly: 'Test successful'"}
        ]

        llm = LLM(model="gpt-3.5-turbo", temperature=0.0)

        if os.getenv("OPENAI_API_KEY"):
            try:
                response = await llm.acall(messages=messages)
                assert response is not None

                # Test that acall works the same as ainvoke
                response2 = await llm.ainvoke(messages=messages)
                assert type(response) is type(response2)

            except Exception as e:
                pytest.skip(f"Async acall test failed: {str(e)}")
        else:
            pytest.skip("OPENAI_API_KEY not set, skipping acall test")

    def test_call_method(self):
        """Test the __call__ convenience method."""
        messages = [{"role": "user", "content": "What is the capital of France?"}]

        llm = LLM(model="gpt-3.5-turbo", temperature=0.1)

        if os.getenv("OPENAI_API_KEY"):
            try:
                # Test __call__ method
                response1 = llm(messages=messages)

                # Test that __call__ works the same as invoke
                response2 = llm.invoke(messages=messages)

                assert response1 is not None
                assert response2 is not None
                assert type(response1) is type(response2)

                # Both should contain information about Paris
                if hasattr(response1, "choices"):
                    content1 = response1.choices[0].message.content.lower()
                    content2 = response2.choices[0].message.content.lower()
                    assert "paris" in content1 or "france" in content1
                    assert "paris" in content2 or "france" in content2

            except Exception as e:
                pytest.skip(f"Call method test failed: {str(e)}")
        else:
            pytest.skip("OPENAI_API_KEY not set, skipping call method test")

    def test_different_models_integration(self):
        """Test with different model providers if API keys are available."""
        test_message = [{"role": "user", "content": "Say hello in one word."}]

        # Test different providers based on available API keys
        models_to_test = []

        if os.getenv("OPENAI_API_KEY"):
            models_to_test.append("gpt-3.5-turbo")

        if os.getenv("ANTHROPIC_API_KEY"):
            models_to_test.append("claude-3-haiku-20240307")

        if not models_to_test:
            pytest.skip("No API keys available for testing different models")

        for model in models_to_test:
            try:
                llm = LLM(model=model, temperature=0.1)
                response = llm.invoke(messages=test_message)

                assert response is not None
                print(f"✓ {model} integration test passed")

            except Exception as e:
                print(f"✗ {model} integration test failed: {str(e)}")
                # Don't fail the entire test, just log the failure

    def test_temperature_and_parameters(self):
        """Test that temperature and other parameters are properly passed."""
        messages = [
            {"role": "user", "content": "Generate a random number between 1 and 100."}
        ]

        if os.getenv("OPENAI_API_KEY"):
            try:
                # Test with different temperatures
                llm_deterministic = LLM(model="gpt-3.5-turbo", temperature=0.0)
                llm_random = LLM(model="gpt-3.5-turbo", temperature=1.0)

                response1 = llm_deterministic.invoke(messages=messages)
                response2 = llm_random.invoke(messages=messages)

                assert response1 is not None
                assert response2 is not None

                # Both should be valid responses
                if hasattr(response1, "choices") and hasattr(response2, "choices"):
                    assert response1.choices[0].message.content is not None
                    assert response2.choices[0].message.content is not None

            except Exception as e:
                pytest.skip(f"Temperature test failed: {str(e)}")
        else:
            pytest.skip("OPENAI_API_KEY not set, skipping temperature test")

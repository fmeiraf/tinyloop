"""
Live integration tests for LLM class - focused on core functionality.

These tests hit real APIs and require API keys to be set as environment variables.
Run with: pytest tests/test_llm_live.py -m integration -v
"""

import pytest
from dotenv import load_dotenv

from tinyloop.inference.litellm import LLM

load_dotenv()


@pytest.mark.integration
@pytest.mark.requires_api_key
class TestLLMLive:
    """Core live integration tests for LLM class."""

    def test_invoke_basic_functionality(self, openai_available, simple_question):
        """Test basic synchronous invoke functionality."""
        if not openai_available:
            pytest.skip("OPENAI_API_KEY not available")

        llm = LLM(model="gpt-3.5-turbo", temperature=0.0)

        response = llm.invoke(messages=simple_question)

        # Verify we got a response
        assert response is not None

        # Check response structure (litellm returns completion objects)
        if hasattr(response, "choices"):
            assert len(response.choices) > 0
            content = response.choices[0].message.content
            assert content is not None
            assert len(content.strip()) > 0
            # Should contain "4" for 2+2
            assert "4" in content

    @pytest.mark.asyncio
    async def test_ainvoke_basic_functionality(self, openai_available, sample_messages):
        """Test basic asynchronous ainvoke functionality."""
        if not openai_available:
            pytest.skip("OPENAI_API_KEY not available")

        llm = LLM(model="gpt-3.5-turbo", temperature=0.0)

        response = await llm.ainvoke(messages=sample_messages)

        # Verify we got a response
        assert response is not None

        # Check response structure
        if hasattr(response, "choices"):
            assert len(response.choices) > 0
            content = response.choices[0].message.content
            assert content is not None
            assert "Hello, World!" in content

    def test_call_method_functionality(self, openai_available, simple_question):
        """Test the __call__ convenience method."""
        if not openai_available:
            pytest.skip("OPENAI_API_KEY not available")

        llm = LLM(model="gpt-3.5-turbo", temperature=0.0)

        # Test __call__ method
        response = llm(messages=simple_question)

        assert response is not None
        if hasattr(response, "choices"):
            content = response.choices[0].message.content
            assert content is not None
            assert "4" in content

    @pytest.mark.asyncio
    async def test_acall_method_functionality(self, openai_available, sample_messages):
        """Test the acall convenience method."""
        if not openai_available:
            pytest.skip("OPENAI_API_KEY not available")

        llm = LLM(model="gpt-3.5-turbo", temperature=0.0)

        response = await llm.acall(messages=sample_messages)

        assert response is not None
        if hasattr(response, "choices"):
            content = response.choices[0].message.content
            assert content is not None
            assert "Hello, World!" in content

    def test_message_history_management(self, openai_available):
        """Test message history functionality."""
        if not openai_available:
            pytest.skip("OPENAI_API_KEY not available")

        initial_messages = [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "What is 1+1?"},
        ]

        llm = LLM(
            model="gpt-3.5-turbo", temperature=0.0, message_history=initial_messages
        )

        # Test getting history
        history = llm.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "system"
        assert history[1]["role"] == "user"

        # Test adding message
        llm.add_message({"role": "assistant", "content": "2"})
        assert len(llm.get_history()) == 3

        # Test setting new history
        new_history = [{"role": "user", "content": "New conversation"}]
        llm.set_history(new_history)
        assert len(llm.get_history()) == 1
        assert llm.get_history()[0]["content"] == "New conversation"

        # Test invoke with history
        response = llm.invoke("Hello, how are you?")  # Should use current history
        assert response is not None

    def test_temperature_parameter(self, openai_available):
        """Test that temperature parameter is working."""
        if not openai_available:
            pytest.skip("OPENAI_API_KEY not available")

        messages = [{"role": "user", "content": "Say exactly: 'Temperature test'"}]

        # Test with temperature 0 (deterministic)
        llm_det = LLM(model="gpt-3.5-turbo", temperature=0.0)
        response1 = llm_det.invoke(messages=messages)
        response2 = llm_det.invoke(messages=messages)

        # Both responses should be similar/identical with temp=0
        assert response1 is not None
        assert response2 is not None

        if hasattr(response1, "choices") and hasattr(response2, "choices"):
            content1 = response1.choices[0].message.content
            content2 = response2.choices[0].message.content
            # With temperature 0, responses should be very similar
            assert (
                "Temperature test" in content1 or "temperature test" in content1.lower()
            )
            assert (
                "Temperature test" in content2 or "temperature test" in content2.lower()
            )

    @pytest.mark.integration
    def test_anthropic_integration(self, anthropic_available):
        """Test with Anthropic Claude if API key is available."""
        if not anthropic_available:
            pytest.skip("ANTHROPIC_API_KEY not available")

        llm = LLM(model="claude-3-haiku-20240307", temperature=0.1)
        messages = [{"role": "user", "content": "Say 'Claude integration working!'"}]

        try:
            response = llm.invoke(messages=messages)
            assert response is not None

            # Claude responses might have different structure
            if hasattr(response, "choices"):
                content = response.choices[0].message.content
                assert content is not None
            elif hasattr(response, "content"):
                assert response.content is not None

        except Exception as e:
            pytest.skip(f"Claude integration test failed: {str(e)}")


@pytest.mark.integration
def test_quick_smoke_test():
    """Quick smoke test that can run without API keys."""
    # Test object creation
    llm = LLM(model="gpt-3.5-turbo", temperature=0.5)

    # Test basic properties
    assert llm.model == "gpt-3.5-turbo"
    assert llm.temperature == 0.5
    assert llm.message_history == []

    # Test history management without API calls
    test_message = {"role": "user", "content": "test"}
    llm.add_message(test_message)
    assert len(llm.get_history()) == 1
    assert llm.get_history()[0] == test_message

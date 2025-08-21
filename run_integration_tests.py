#!/usr/bin/env python3
"""
Quick script to run integration tests and verify setup.
Run with: uv run python run_integration_tests.py
"""

import os
import sys


def check_api_keys():
    """Check which API keys are available."""
    keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    }

    print("API Key Status:")
    print("-" * 20)
    for key, value in keys.items():
        status = "✓ Set" if value else "✗ Not set"
        print(f"{key}: {status}")

    return any(keys.values())


def run_smoke_test():
    """Run a basic smoke test without API calls."""
    try:
        from tinyloop.inference.litellm import LLM

        print("\nRunning smoke test...")
        llm = LLM(model="gpt-3.5-turbo", temperature=0.5)

        # Test basic functionality
        assert llm.model == "gpt-3.5-turbo"
        assert llm.temperature == 0.5
        assert llm.message_history == []

        # Test history management
        test_message = {"role": "user", "content": "test"}
        llm.add_message(test_message)
        assert len(llm.get_history()) == 1

        print("✓ Smoke test passed!")
        return True

    except Exception as e:
        print(f"✗ Smoke test failed: {e}")
        return False


def run_live_test():
    """Run a live test if API key is available."""
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping live test - no OPENAI_API_KEY")
        return True

    try:
        from tinyloop.inference.litellm import LLM

        print("\nRunning live integration test...")
        llm = LLM(model="gpt-3.5-turbo", temperature=0.0)

        messages = [{"role": "user", "content": "Say 'Integration test successful!'"}]
        response = llm.invoke(messages=messages)

        if response and hasattr(response, "choices"):
            content = response.choices[0].message.content
            print(f"Response: {content}")
            print("✓ Live test passed!")
            return True
        else:
            print("✗ Unexpected response format")
            return False

    except Exception as e:
        print(f"✗ Live test failed: {e}")
        return False


def main():
    """Main function."""
    print("TinyLoop Integration Test Runner")
    print("=" * 40)

    # Check API keys
    has_keys = check_api_keys()

    # Run smoke test
    smoke_passed = run_smoke_test()

    # Run live test if possible
    live_passed = True
    if has_keys:
        live_passed = run_live_test()

    # Summary
    print("\nTest Summary:")
    print("-" * 20)
    print(f"Smoke test: {'✓ Passed' if smoke_passed else '✗ Failed'}")
    print(f"Live test: {'✓ Passed' if live_passed else '✗ Failed'}")

    if smoke_passed and live_passed:
        print("\n🎉 All tests passed! Integration tests are ready to use.")
        print("\nTo run full test suite:")
        print("  pytest tests/test_llm_live.py -v")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

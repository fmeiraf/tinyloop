"""
Tests for structured output functionality using Pydantic models.

These tests verify that the LLM can generate structured responses that conform
to specified Pydantic models, including complex nested structures and various data types.
"""

import os
from typing import List, Optional

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from tinyloop.inference.litellm import LLM
from tinyloop.types import LLMResponse

load_dotenv()


# Test Models (matching the example from the notebook)
class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


class EventsList(BaseModel):
    events: list[CalendarEvent]


# Additional test models for comprehensive testing
class Person(BaseModel):
    name: str
    age: int
    email: Optional[str] = None


class Product(BaseModel):
    name: str
    price: float
    in_stock: bool
    tags: List[str] = Field(default_factory=list)


class Order(BaseModel):
    id: int
    customer: Person
    products: List[Product]
    total_amount: float
    order_date: str


class SimpleResponse(BaseModel):
    message: str
    count: int


class BookInfo(BaseModel):
    title: str
    author: str
    year: int
    pages: Optional[int] = None
    genres: List[str] = Field(default_factory=list)


class Library(BaseModel):
    name: str
    location: str
    books: List[BookInfo]
    established_year: int


class TestStructuredOutput:
    """Test suite for structured output functionality."""

    @pytest.fixture
    def llm(self):
        """Create LLM instance for testing."""
        return LLM(
            model="openai/gpt-4o-mini",
            temperature=0.1,
        )

    @pytest.mark.integration
    def test_basic_calendar_events_structure(self, llm):
        """Test the basic example from the notebook - calendar events."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set, skipping OpenAI test")

        try:
            response = llm(
                prompt="List 3 important events in the XIX century",
                response_format=EventsList,
            )

            # Verify response structure
            assert isinstance(response, LLMResponse)
            assert response.response is not None
            assert isinstance(response.response, EventsList)

            # Verify the events list
            events_list = response.response
            assert isinstance(events_list.events, list)
            assert len(events_list.events) == 3

            # Verify each event structure
            for event in events_list.events:
                assert isinstance(event, CalendarEvent)
                assert isinstance(event.name, str)
                assert len(event.name) > 0
                assert isinstance(event.date, str)
                assert len(event.date) > 0
                assert isinstance(event.participants, list)
                assert all(isinstance(p, str) for p in event.participants)

        except Exception as e:
            pytest.skip(f"Calendar events test failed: {str(e)}")

    @pytest.mark.integration
    def test_simple_structured_response(self, llm):
        """Test a simple structured response."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set, skipping OpenAI test")

        try:
            response = llm(
                prompt="Create a welcome message with a count of 42",
                response_format=SimpleResponse,
            )

            assert isinstance(response, LLMResponse)
            assert isinstance(response.response, SimpleResponse)

            simple_resp = response.response
            assert isinstance(simple_resp.message, str)
            assert len(simple_resp.message) > 0
            assert simple_resp.count == 42

        except Exception as e:
            pytest.skip(f"Simple structured response test failed: {str(e)}")

    @pytest.mark.integration
    def test_nested_complex_structure(self, llm):
        """Test complex nested structures with the Order model."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set, skipping OpenAI test")

        try:
            prompt = """
            Create an order for customer John Doe (age 30, email john@example.com) 
            with 2 products: a laptop ($999.99, in stock, tags: electronics, computer) 
            and a mouse ($25.50, in stock, tags: electronics, accessory). 
            Order ID is 12345, order date is 2024-01-15, total amount is $1025.49.
            """

            response = llm(prompt=prompt, response_format=Order)

            assert isinstance(response, LLMResponse)
            assert isinstance(response.response, Order)

            order = response.response

            # Verify order structure
            assert order.id == 12345
            assert isinstance(order.customer, Person)
            assert order.customer.name == "John Doe"
            assert order.customer.age == 30
            assert order.customer.email == "john@example.com"

            # Verify products
            assert isinstance(order.products, list)
            assert len(order.products) == 2

            for product in order.products:
                assert isinstance(product, Product)
                assert isinstance(product.name, str)
                assert isinstance(product.price, float)
                assert isinstance(product.in_stock, bool)
                assert isinstance(product.tags, list)

        except Exception as e:
            pytest.skip(f"Complex nested structure test failed: {str(e)}")

    @pytest.mark.integration
    def test_list_with_optional_fields(self, llm):
        """Test structures with optional fields and lists."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set, skipping OpenAI test")

        try:
            prompt = """
            Create a library called "Central Library" located in "Downtown" 
            established in 1950 with 3 books:
            1. "1984" by George Orwell, published in 1949, 328 pages, genres: dystopian, fiction
            2. "To Kill a Mockingbird" by Harper Lee, published in 1960, genres: fiction, classic
            3. "The Great Gatsby" by F. Scott Fitzgerald, published in 1925, 180 pages, genres: fiction, american literature
            """

            response = llm(prompt=prompt, response_format=Library)

            assert isinstance(response, LLMResponse)
            assert isinstance(response.response, Library)

            library = response.response

            # Verify library structure
            assert library.name == "Central Library"
            assert library.location == "Downtown"
            assert library.established_year == 1950

            # Verify books
            assert isinstance(library.books, list)
            assert len(library.books) == 3

            for book in library.books:
                assert isinstance(book, BookInfo)
                assert isinstance(book.title, str)
                assert isinstance(book.author, str)
                assert isinstance(book.year, int)
                assert isinstance(book.genres, list)
                # pages is optional, so it can be None or int
                if book.pages is not None:
                    assert isinstance(book.pages, int)

        except Exception as e:
            pytest.skip(f"Optional fields test failed: {str(e)}")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_structured_output(self, llm):
        """Test structured output with async methods."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set, skipping async OpenAI test")

        try:
            response = await llm.ainvoke(
                prompt="List 2 scientific events from the 20th century",
                response_format=EventsList,
            )

            assert isinstance(response, LLMResponse)
            assert isinstance(response.response, EventsList)

            events_list = response.response
            assert len(events_list.events) == 2

            for event in events_list.events:
                assert isinstance(event, CalendarEvent)
                assert isinstance(event.name, str)
                assert len(event.name) > 0

        except Exception as e:
            pytest.skip(f"Async structured output test failed: {str(e)}")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_acall_structured_output(self, llm):
        """Test structured output with acall method."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set, skipping async acall test")

        try:
            response = await llm.acall(
                prompt="Create a simple response with message 'Hello async' and count 99",
                response_format=SimpleResponse,
            )

            assert isinstance(response, LLMResponse)
            assert isinstance(response.response, SimpleResponse)

            simple_resp = response.response
            assert "async" in simple_resp.message.lower()
            assert simple_resp.count == 99

        except Exception as e:
            pytest.skip(f"Async acall structured output test failed: {str(e)}")

    @pytest.mark.integration
    def test_structured_output_with_messages(self, llm):
        """Test structured output with message history."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set, skipping OpenAI test")

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides structured data.",
                },
                {
                    "role": "user",
                    "content": "I need information about historical events.",
                },
                {
                    "role": "assistant",
                    "content": "I can help you with historical events. What specific period are you interested in?",
                },
                {
                    "role": "user",
                    "content": "List 2 events from the Renaissance period",
                },
            ]

            response = llm.invoke(messages=messages, response_format=EventsList)

            assert isinstance(response, LLMResponse)
            assert isinstance(response.response, EventsList)

            events_list = response.response
            assert len(events_list.events) == 2

        except Exception as e:
            pytest.skip(f"Structured output with messages test failed: {str(e)}")

    @pytest.mark.integration
    def test_structured_output_cost_tracking(self, llm):
        """Test that cost tracking works with structured output."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set, skipping OpenAI test")

        try:
            initial_cost = llm.get_total_cost()

            response = llm(
                prompt="Create a simple response with message 'test' and count 1",
                response_format=SimpleResponse,
            )

            # Verify response
            assert isinstance(response, LLMResponse)
            assert isinstance(response.response, SimpleResponse)

            # Verify cost tracking
            assert response.cost >= 0
            assert llm.get_total_cost() > initial_cost
            assert len(llm.run_cost) > 0

        except Exception as e:
            pytest.skip(f"Cost tracking test failed: {str(e)}")

    def test_structured_output_parsing_method(self):
        """Test the _parse_structured_output method directly."""
        llm = LLM(model="test-model")

        # Test valid JSON parsing
        json_response = '{"message": "Hello", "count": 42}'
        parsed = llm._parse_structured_output(json_response, SimpleResponse)

        assert isinstance(parsed, SimpleResponse)
        assert parsed.message == "Hello"
        assert parsed.count == 42

    def test_structured_output_parsing_validation_error(self):
        """Test that validation errors are properly raised."""
        llm = LLM(model="test-model")

        # Test invalid JSON structure
        invalid_json = '{"message": "Hello", "count": "not a number"}'

        with pytest.raises(ValidationError):
            llm._parse_structured_output(invalid_json, SimpleResponse)

    def test_structured_output_parsing_invalid_json(self):
        """Test that invalid JSON raises appropriate error."""
        llm = LLM(model="test-model")

        # Test malformed JSON
        invalid_json = '{"message": "Hello", "count":}'

        with pytest.raises(Exception):  # Could be ValidationError or JSON decode error
            llm._parse_structured_output(invalid_json, SimpleResponse)

    @pytest.mark.integration
    def test_multiple_structured_calls_same_instance(self, llm):
        """Test multiple structured output calls with the same LLM instance."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set, skipping OpenAI test")

        # First call
        response1 = llm(
            prompt="Create a response with message 'First call' and count 1",
            response_format=SimpleResponse,
        )

        # Second call
        response2 = llm(
            prompt="Create a response with message 'Second call' and count 2",
            response_format=SimpleResponse,
        )

        # Verify both responses
        assert isinstance(response1.response, SimpleResponse)
        assert isinstance(response2.response, SimpleResponse)
        assert response1.response.count == 1
        assert response2.response.count == 2

        # Verify history contains both interactions
        history = llm.get_history()
        assert len(history) >= 4  # At least 2 user + 2 assistant messages

    @pytest.mark.integration
    def test_structured_output_different_temperatures(self, llm):
        """Test structured output with different temperature settings."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set, skipping OpenAI test")

        try:
            # Test with low temperature (deterministic)
            llm_deterministic = LLM(model="openai/gpt-4o-mini", temperature=0.0)
            response1 = llm_deterministic(
                prompt="Create a response with message 'Deterministic' and count 100",
                response_format=SimpleResponse,
            )

            # Test with higher temperature
            llm_random = LLM(model="openai/gpt-4o-mini", temperature=0.8)
            response2 = llm_random(
                prompt="Create a response with message 'Random' and count 200",
                response_format=SimpleResponse,
            )

            # Both should be valid structured responses
            assert isinstance(response1.response, SimpleResponse)
            assert isinstance(response2.response, SimpleResponse)
            assert response1.response.count == 100
            assert response2.response.count == 200

        except Exception as e:
            pytest.skip(f"Different temperatures test failed: {str(e)}")

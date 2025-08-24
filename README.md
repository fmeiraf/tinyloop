<p align="center">
  <img src="docs/images/tiny_logo_v1.png" alt="tinyLoop Logo" width="200"/>
</p>

> A lightweight Python library for building AI-powered applications with clean function calling, vision support, and MLflow integration.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/pypi-tinyloop-blue.svg)](https://pypi.org/project/tinyloop/)

TinyLoop is fully built on top of [LiteLLM](https://github.com/BerriAI/litellm), providing 100% compatibility with the LiteLLM API while adding powerful abstractions and utilities. This means you can use any model, provider, or feature that LiteLLM supports, including:

- **All LLM Providers**: OpenAI, Anthropic, Google, Azure, Cohere, and 100+ more
- **All Model Types**: Chat, completion, embedding, and vision models
- **Advanced Features**: Streaming, function calling, structured outputs, and more
- **Ops Features**: Retries, fallbacks, caching, and cost tracking

TinyLoop provides a clean, intuitive interface for working with Large Language Models (LLMs), featuring:

- 🎯 **Clean Function Calling**: Convert Python functions to JSON tool definitions automatically
- 🔍 **MLflow Integration**: Built-in tracing and monitoring with customizable span names
- 👁️ **Vision Support**: Handle images and vision models seamlessly
- 📊 **Structured Output**: Generate structured data from LLM responses using Pydantic
- 🔄 **Tool Loops**: Execute multi-step tool calling workflows
- ⚡ **Async Support**: Full async/await support for all operations

## 📦 Installation

```bash
pip install tinyloop
```

## 🚀 Quick Start

### Basic LLM Usage

#### Synchronous Calls

```python
from tinyloop.inference.litellm import LLM

# Initialize the LLM
llm = LLM(model="openai/gpt-3.5-turbo", temperature=0.1)

# Simple text generation
response = llm(prompt="Hello, how are you?")
print(response)

# Get conversation history
history = llm.get_history()

# Access comprehensive response information
print(f"Response: {response}")
print(f"Cost: ${response.cost:.6f}")
print(f"Tool calls: {response.tool_calls}")
print(f"Raw response: {response.raw_response}")
print(f"Message history: {len(response.message_history)} messages")
```

#### Asynchronous Calls

```python
from tinyloop.inference.litellm import LLM

llm = LLM(model="openai/gpt-3.5-turbo", temperature=0.1)

# Async text generation
response = await llm.acall(prompt="Hello, how are you?")
print(response)
```

### 🎯 Structured Output Generation

Generate structured data using Pydantic models:

```python
from tinyloop.inference.litellm import LLM
from pydantic import BaseModel
from typing import List

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: List[str]

class EventsList(BaseModel):
    events: List[CalendarEvent]

# Initialize LLM with structured output
llm = LLM(
    model="openai/gpt-4.1-nano",
    temperature=0.1,
)

# Generate structured data
response = llm(
    prompt="List 5 important events in the XIX century",
    response_format=EventsList
)

# Access structured data
for event in response.events:
    print(f"{event.name} - {event.date}")
    print(f"Participants: {', '.join(event.participants)}")
```

### 📊 Comprehensive Response Information

Every TinyLoop request returns a rich response object with detailed information:

```python
from tinyloop.inference.litellm import LLM

llm = LLM(model="openai/gpt-4.1-nano", temperature=0.1)

response = llm(prompt="Explain quantum computing in one sentence")

# Access the main response content
print(f"Response: {response}")

# Cost tracking (in USD)
print(f"Cost: ${response.cost:.6f}")

# Tool calls (if any were made)
print(f"Tool calls: {response.tool_calls}")

# Raw LiteLLM response object
print(f"Raw response: {response.raw_response}")

# Complete conversation history
print(f"Message history: {len(response.message_history)} messages")

# Hidden fields with additional metadata
print(f"Provider: {response.hidden_fields.get('custom_llm_provider')}")
print(f"Model: {response.hidden_fields.get('litellm_model_name')}")
print(f"Response time: {response.hidden_fields.get('_response_ms')}ms")
```

### 👁️ Vision Support

Work with images using various input methods:

```python
from tinyloop.inference.litellm import LLM
from tinyloop.features.vision import Image
from PIL import Image as PILImage

llm = LLM(model="openai/gpt-4.1-nano", temperature=0.1)

# From PIL Image
pil_image = PILImage.open("image.jpg")
image = Image.from_PIL(pil_image)

# From file path
image = Image.from_file("image.jpg")

# From URL
image = Image.from_url("https://example.com/image.jpg")

# Analyze image
response = llm(prompt="Describe this image", images=[image])
print(response)
```

### 🔧 Function Calling

Convert Python functions to LLM tools with automatic schema generation:

```python
from tinyloop.inference.litellm import LLM
from tinyloop.features.function_calling import Tool
import json

def get_current_weather(location: str, unit: str):
    """Get the current weather in a given location

    Args:
        location: The city and state, e.g. San Francisco, CA
        unit: Temperature unit {'celsius', 'fahrenheit'}

    Returns:
        A sentence indicating the weather
    """
    if location == "Boston, MA":
        return "The weather is 12°F"
    return f"Weather in {location} is sunny"

# Create LLM instance
llm = LLM(model="openai/gpt-4.1-nano", temperature=0.1)

# Create tool from function
weather_tool = Tool(get_current_weather)

# Use function calling
inference = llm(
    prompt="What is the weather in Boston, MA?",
    tools=[weather_tool],
)

# Process tool calls
for tool_call in inference.raw_response.choices[0].message.tool_calls:
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)
    print(f"Tool: {tool_name}")
    print(f"Args: {tool_args}")
    print(weather_tool(**tool_args))

# Access comprehensive response information
print(f"Total cost: ${inference.cost:.6f}")
print(f"Tool calls made: {len(inference.tool_calls) if inference.tool_calls else 0}")
print(f"Conversation length: {len(inference.message_history)} messages")
```

### 🔄 Tool Loops

Execute multi-step tool calling workflows:

```python
from tinyloop.modules.tool_loop import ToolLoop
from tinyloop.features.function_calling import Tool
from pydantic import BaseModel
import random

def roll_dice():
    """Roll a dice and return the result"""
    return random.randint(1, 6)

class FinalAnswer(BaseModel):
    last_roll: int
    reached_goal: bool

# Create tool loop
loop = ToolLoop(
    model="openai/gpt-4.1",
    system_prompt="""
    You are a dice rolling assistant.
    Roll a dice until you get the number indicated in the prompt.
    Use the roll_dice function to roll the dice.
    Return the last roll and whether you reached the goal.
    """,
    temperature=0.1,
    output_format=FinalAnswer,
    tools=[Tool(roll_dice)]
)

# Execute the loop
response = loop(
    prompt="Roll a dice until you get a 6",
    parallel_tool_calls=False,
)

print(f"Last roll: {response.last_roll}")
print(f"Reached goal: {response.reached_goal}")
```

### 📊 Generate Module

Use the dedicated Generate module for structured content generation:

```python
from tinyloop.modules.generate import Generate
from pydantic import BaseModel
from typing import List

class Character(BaseModel):
    name: str
    description: str
    image: str

class Characters(BaseModel):
    characters: List[Character]

# Create generate instance
generate = Generate(
    model="openai/gpt-4.1-nano",
    temperature=0.1,
    output_format=Characters
)

# Generate structured content
response = generate(prompt="Give me 3 Harry Potter characters")

for character in response.characters:
    print(f"Name: {character.name}")
    print(f"Description: {character.description}")
    print(f"Image: {character.image}")
    print("---")
```

## 🔍 MLflow Integration

### Automatic Tracing

TinyLoop automatically integrates with MLflow for tracing:

```python
import mlflow
from tinyloop.features.function_calling import Tool

def get_stock_price(symbol: str, currency: str = "USD"):
    """Get stock price for a symbol."""
    return f"Stock price for {symbol}: $150.00 {currency}"

# Create tool with custom name
stock_tool = Tool(get_stock_price, name="stock_service")

# Start MLflow run
with mlflow.start_run():
    # This will create a span named "stock_service.__call__"
    result = stock_tool("AAPL", "USD")
```

### Custom Tracing

For advanced scenarios, use custom MLflow tracing:

```python
from tinyloop.utils.mlflow import mlflow_trace_custom
import mlflow

class CustomTool:
    def __init__(self, name: str):
        self.name = name

    @mlflow_trace_custom(
        mlflow.entities.SpanType.TOOL,
        lambda self, func: f"{self.name}.{func.__name__}"
    )
    def execute(self, *args, **kwargs):
        return f"Executed {self.name} with {args}"

# This will create spans named "my_tool.execute"
tool = CustomTool("my_tool")
```

## 🏗️ Project Structure

```
tinyloop/
├── features/
│   ├── function_calling.py  # Function calling utilities
│   └── vision.py           # Vision model support
├── inference/
│   ├── base.py             # Base inference classes
│   └── litellm.py          # LiteLLM integration
├── modules/
│   ├── base_loop.py        # Base loop implementation
│   ├── generate.py         # Generation modules
│   └── tool_loop.py        # Tool execution loop
└── utils/
    └── mlflow.py           # MLflow utilities
```

## 🧪 Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_function_calling.py -v

# Run with coverage
pytest tests/ --cov=tinyloop
```

### Examples

Check out the Jupyter notebooks for more detailed examples:

- [`basic_usage.ipynb`](notebooks/basic_usage.ipynb) - Basic usage examples
- [`modules.ipynb`](notebooks/modules.ipynb) - Advanced module usage

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Made with ❤️ for the AI community
</div>

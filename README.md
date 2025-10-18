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

### Supported Features

#### 🎯 Structured Output Generation

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

#### 👁️ Vision

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

#### 🔧 Function Calling

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

### 📝 Generate Module

Simple text generation with a clean interface:

```python
from tinyloop.modules.generate import Generate

# Synchronous generation
response = Generate.run(
    prompt="Write a haiku about programming",
    model="openai/gpt-3.5-turbo",
    temperature=0.7
)
print(response.response)

# Async generation
response = await Generate.arun(
    prompt="Explain quantum computing",
    model="openai/gpt-4",
    temperature=0.3
)
print(response.response)

# Using the class for multiple calls
generator = Generate(
    model="openai/gpt-3.5-turbo",
    temperature=0.5,
    system_prompt="You are a helpful coding assistant."
)

response1 = generator.call("How do I implement a binary search?")
response2 = generator.call("What's the time complexity?")
```

### 🎨 Prompt Rendering

Manage prompts with YAML templates and Jinja2:

```python
from tinyloop.utils.prompt_renderer import PromptRenderer, render_base_prompts

# Using PromptRenderer class
renderer = PromptRenderer("prompts/chat.yaml")
system_prompt = renderer.render("system", user_name="Alice", context="coding")
user_prompt = renderer.render("user", question="How do I debug Python?")

```

**Example YAML prompt file (`prompts/chat.yaml`):**

```yaml
system: |
  You are {{ user_name }}, a helpful AI assistant specializing in {{ context }}.
  Always provide clear, actionable advice.

user: |
  {{ user_name }}, I have a question: {{ question }}

  Please provide a detailed response with examples if relevant.
```

### 🌊 Streaming Responses

Get real-time responses as they're generated:

```python
from tinyloop.inference.litellm import LLM

llm = LLM(model="openai/gpt-3.5-turbo", temperature=0.1)

# Stream responses
for chunk in llm.stream(prompt="Write a story about a robot"):
    print(chunk.response, end="", flush=True)
```

### 🔄 Async Tool Loops

Execute tool loops asynchronously for better performance:

```python
import asyncio
from tinyloop.modules.tool_loop import ToolLoop
from tinyloop.features.function_calling import Tool
from pydantic import BaseModel

def fetch_data(source: str):
    """Fetch data from a source"""
    return f"Data from {source}: [1, 2, 3, 4, 5]"

def process_data(data: str):
    """Process the fetched data"""
    return f"Processed: {data}"

class AnalysisResult(BaseModel):
    final_result: str
    steps_completed: int

async def main():
    loop = ToolLoop(
        model="openai/gpt-4",
        system_prompt="You are a data analyst. Fetch and process data step by step.",
        temperature=0.1,
        output_format=AnalysisResult,
        tools=[Tool(fetch_data), Tool(process_data)]
    )

    result = await loop.acall(
        prompt="Fetch data from 'api' and process it"
    )
    print(f"Final result: {result.final_result}")
    print(f"Steps completed: {result.steps_completed}")

# Run the async function
asyncio.run(main())
```

### 🔍 Advanced Observability: MLflow Integration

#### Custom Span Names

Create custom MLflow spans with meaningful names:

```python
from tinyloop.utils.mlflow import mlflow_trace
from tinyloop.features.function_calling import Tool
import mlflow

def get_weather(location: str, unit: str = "celsius"):
    """Get weather for a location"""
    return f"Weather in {location}: 20°{unit}"

def get_stock_price(symbol: str, currency: str = "USD"):
    """Get stock price for a symbol"""
    return f"Stock price for {symbol}: $150.00 {currency}"

# Create tools with custom names for better tracing
weather_tool = Tool(get_weather, name="weather_service")
stock_tool = Tool(get_stock_price, name="stock_service")

# Start MLflow run
with mlflow.start_run():
    # Call tools - these will create spans with custom names
    weather_result = weather_tool("London", "fahrenheit")
    stock_result = stock_tool("AAPL", "USD")

    # The MLflow spans will be named:
    # - "weather_service.__call__" for the weather tool
    # - "stock_service.__call__" for the stock tool
```

#### Custom Agent Tracing

```python
from tinyloop.utils.mlflow import mlflow_trace

class ResearchAgent:
    def __init__(self):
        self.llm = LLM(model="openai/gpt-4", temperature=0.1)

    @mlflow_trace(mlflow.entities.SpanType.AGENT)
    def research_topic(self, topic: str):
        """Research a topic comprehensively"""
        response = self.llm(
            prompt=f"Research the topic: {topic}. Provide key insights and sources."
        )
        return response

    @mlflow_trace(mlflow.entities.SpanType.AGENT)
    def analyze_findings(self, findings: str):
        """Analyze research findings"""
        response = self.llm(
            prompt=f"Analyze these findings: {findings}. What are the implications?"
        )
        return response

# Usage with automatic tracing
agent = ResearchAgent()
research_result = agent.research_topic("artificial intelligence")
analysis_result = agent.analyze_findings(research_result.response)
```

### 🛡️ Error Handling and Retries

Handle errors gracefully with retry patterns:

```python
from tinyloop.inference.litellm import LLM
import time
import random

def robust_llm_call(llm, prompt, max_retries=3, delay=1):
    """Make LLM calls with retry logic"""
    for attempt in range(max_retries):
        try:
            response = llm(prompt=prompt)
            return response
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay * (2 ** attempt) + random.uniform(0, 1))

    return None

# Usage
llm = LLM(model="openai/gpt-3.5-turbo", temperature=0.1)
response = robust_llm_call(
    llm,
    "Explain the concept of machine learning",
    max_retries=3
)
print(response.response)
```

### 🔍 Observability: MLflow Integration

#### Automatic Tracing

TinyLoop automatically integrates with MLflow for tracing:

```python
from tinyloop.utils.mlflow import mlflow_trace

class Agent:
    @mlflow_trace(mlflow.entities.SpanType.AGENT)
    def __call__(self, prompt: str, **kwargs):
        self.llm.add_message(self.llm._prepare_user_message(prompt))
        for _ in range(self.max_iterations):
            response = self.llm(
                messages=self.llm.get_history(), tools=self.tools, **kwargs
            )
            if response.tool_calls:
                should_finish = False
                for tool_call in response.tool_calls:
                    tool_response = self.tools_map[tool_call.function_name](
                        **tool_call.args
                    )

                    self.llm.add_message(
                        self._format_tool_response(tool_call, str(tool_response))
                    )

                    if tool_call.function_name == "finish":
                        should_finish = True
                        break

                if should_finish:
                    break

        return self.llm(
            messages=self.llm.get_history(),
            response_format=self.output_format,
    )
```

<p align="center">
  <img src="docs/images/mlflow_example.png" alt="tinyLoop Logo"/>
</p>

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

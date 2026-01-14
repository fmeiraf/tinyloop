# TinyLoop CTX Specification

## Overview

**Feature Name:** CTX (Context Analyzer)
**Command:** `tinyloop ctx`
**Version:** 1.0.0
**Status:** Draft

CTX is a diagnostic tool for analyzing LLM conversation token usage and detecting when conversations enter the "dumb zone" - a region of the context window where model performance may degrade.

## Background

### The Dumb Zone Concept

Research and practical experience suggest that LLM performance can degrade when the context window fills beyond a certain threshold. The "dumb zone" refers to the portion of the context window (default: after 40% utilization) where:

- Information retrieval accuracy may decrease
- The model may "forget" or deprioritize earlier context
- Response quality may become inconsistent

CTX provides visibility into token usage patterns to help developers:
- Diagnose context-related issues
- Optimize conversation design
- Implement proactive context management strategies

## Requirements

### Functional Requirements

#### FR1: CLI Tool
- **FR1.1:** Provide a `tinyloop ctx` command for analyzing conversations
- **FR1.2:** Accept input from JSON file path argument
- **FR1.3:** Accept input from stdin (piped JSON)
- **FR1.4:** Support two output modes:
  - **Full report:** Formatted TUI table with round-by-round breakdown
  - **Simple output:** One-line status with current tokens, threshold, and distance to dumb zone

#### FR2: Token Counting
- **FR2.1:** Count tokens using provider-appropriate tokenizers:
  - Anthropic Claude 3+: Use official `messages.count_tokens()` API
  - OpenAI models: Use `tiktoken` with model-specific encodings
  - Other/unknown models: Use `tiktoken` with `cl100k_base` fallback
- **FR2.2:** Support image token estimation for multi-modal messages
- **FR2.3:** Handle tool definitions, tool calls, and tool responses

#### FR3: Dumb Zone Detection
- **FR3.1:** Calculate dumb zone threshold as percentage of context window
- **FR3.2:** Default threshold: 40% of context window
- **FR3.3:** Default context window: 168,000 tokens
- **FR3.4:** Both values configurable via CLI arguments

#### FR4: Token Categorization
- **FR4.1:** Categorize tokens by message role:
  - `system`: System prompt messages
  - `user`: User messages
  - `assistant`: Assistant responses (text content only)
  - `tools`: Tool definitions, tool calls, and tool responses

#### FR5: Round Tracking
- **FR5.1:** Track token accumulation per "round" (each message = 1 round)
- **FR5.2:** Show cumulative token count progression
- **FR5.3:** Indicate when/where dumb zone threshold is crossed

#### FR6: Programmatic Access (Point-in-Time Analysis)
- **FR6.1:** Provide API to analyze conversation history at any point in time
- **FR6.2:** Support extracting current `message_history` from `LLM` instance
- **FR6.3:** Return both full analysis result and simple status check

#### FR7: LLM Integration (Middleware)
- **FR7.1:** Provide middleware/hook for live `LLM` class sessions
- **FR7.2:** Support configurable behavior when dumb zone is reached:
  - `warn`: Print warning (non-blocking)
  - `raise`: Raise exception
  - Configurable via parameter
- **FR7.3:** Expose method to get current status at any time during conversation

### Non-Functional Requirements

#### NFR1: Dependencies
- Use `typer` for CLI framework
- Use `rich` for TUI table output
- Use `tiktoken` for OpenAI tokenization
- Use `anthropic` SDK for Claude token counting (optional, for accurate counts)

#### NFR2: Performance
- Token counting should complete in <5 seconds for typical conversations (<100 messages)
- Simple status check should complete in <1 second

#### NFR3: Offline Mode
- Support offline estimation using tiktoken fallback when API calls are not possible
- Clearly indicate when estimates are used vs. exact counts

## Architecture

### Module Structure

```
tinyloop/
├── ctx/
│   ├── __init__.py          # Public API exports
│   ├── cli.py               # Typer CLI implementation
│   ├── analyzer.py          # Core analysis logic
│   ├── tokenizers.py        # Tokenizer abstraction layer
│   ├── categories.py        # Token categorization logic
│   └── middleware.py        # LLM class integration
```

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI (typer)                             │
│                       tinyloop ctx                              │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Analyzer                                  │
│  - Orchestrates token counting                                  │
│  - Calculates dumb zone status                                  │
│  - Generates round-by-round report                              │
│  - Provides simple status summary                               │
└─────────────────────────────────────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
┌──────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Tokenizers     │ │   Categories    │ │   Middleware    │
│                  │ │                 │ │                 │
│ - AnthropicAPI   │ │ - system        │ │ - LLM hook      │
│ - Tiktoken       │ │ - user          │ │ - warn/raise    │
│ - Fallback       │ │ - assistant     │ │ - status check  │
└──────────────────┘ │ - tools         │ └─────────────────┘
                     └─────────────────┘
```

## Input Format

### Conversation JSON Schema

The input should be an array of messages following the OpenAI-compatible format used by TinyLoop:

```json
{
  "model": "anthropic/claude-sonnet-4-20250514",
  "context_window": 200000,
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello, how are you?"
    },
    {
      "role": "assistant",
      "content": "I'm doing well, thank you for asking!"
    },
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      ]
    },
    {
      "role": "assistant",
      "content": "I can see...",
      "tool_calls": [
        {
          "id": "call_123",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"location\": \"San Francisco\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_123",
      "content": "{\"temperature\": 72, \"condition\": \"sunny\"}"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          }
        }
      }
    }
  ]
}
```

### Minimal Input

At minimum, only `messages` array is required:

```json
{
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
  ]
}
```

### Direct Message Array

For convenience, a raw message array is also accepted:

```json
[
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

## CLI Interface

### Command Signature

```bash
tinyloop ctx [OPTIONS] [FILE]
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `FILE` | Path (optional) | Path to JSON file. If omitted, reads from stdin |

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--threshold`, `-t` | Float | 0.4 | Dumb zone threshold (0.0-1.0) |
| `--context-window`, `-c` | Integer | 168000 | Context window size in tokens |
| `--model`, `-m` | String | None | Model name for tokenizer selection |
| `--simple`, `-s` | Flag | False | Output simple one-line status instead of full report |
| `--offline` | Flag | False | Use offline estimation only (no API calls) |
| `--verbose`, `-v` | Flag | False | Show detailed token breakdown (full report only) |

### Usage Examples

```bash
# Full report from a conversation file
tinyloop ctx conversation.json

# Simple one-line status
tinyloop ctx -s conversation.json

# Pipe from another command
cat conversation.json | tinyloop ctx

# Simple status from pipe
cat conversation.json | tinyloop ctx --simple

# Custom threshold and context window
tinyloop ctx -t 0.3 -c 200000 conversation.json

# Specify model for accurate tokenization
tinyloop ctx -m anthropic/claude-sonnet-4-20250514 conversation.json

# Offline mode (no API calls)
tinyloop ctx --offline conversation.json

# Verbose output with category breakdown
tinyloop ctx -v conversation.json
```

## Output Formats

### Simple Output (with `--simple` or `-s`)

A single, direct message showing the essential status:

**When safe (not in dumb zone):**
```
✓ 45,230 / 67,200 tokens (26.9%) — 21,970 tokens until dumb zone
```

**When in dumb zone:**
```
⚠ 72,450 / 67,200 tokens (43.1%) — IN DUMB ZONE (5,250 tokens over threshold)
```

**When approaching dumb zone (within 10% of threshold):**
```
⚡ 62,100 / 67,200 tokens (37.0%) — 5,100 tokens until dumb zone (warning: approaching)
```

### Full Report Output (default)

```
╭─────────────────────────────────────────────────────────────────╮
│                     TinyLoop CTX Analysis                       │
├─────────────────────────────────────────────────────────────────┤
│  Model: anthropic/claude-sonnet-4-20250514                      │
│  Context Window: 168,000 tokens                                 │
│  Dumb Zone Threshold: 40% (67,200 tokens)                       │
│  Tokenizer: Anthropic API                                       │
╰─────────────────────────────────────────────────────────────────╯

╭───────┬──────────┬────────┬────────────┬────────┬───────────────╮
│ Round │ Role     │ Tokens │ Cumulative │ % Used │ Status        │
├───────┼──────────┼────────┼────────────┼────────┼───────────────┤
│ 1     │ system   │ 150    │ 150        │ 0.1%   │ ✓ Safe        │
│ 2     │ user     │ 45     │ 195        │ 0.1%   │ ✓ Safe        │
│ 3     │ assistant│ 230    │ 425        │ 0.3%   │ ✓ Safe        │
│ 4     │ user     │ 1,200  │ 1,625      │ 1.0%   │ ✓ Safe        │
│ ...   │ ...      │ ...    │ ...        │ ...    │ ...           │
│ 47    │ assistant│ 3,400  │ 65,200     │ 38.8%  │ ✓ Safe        │
│ 48    │ user     │ 890    │ 66,090     │ 39.3%  │ ✓ Safe        │
│ 49    │ assistant│ 2,100  │ 68,190     │ 40.6%  │ ⚠ DUMB ZONE   │
│ 50    │ user     │ 450    │ 68,640     │ 40.9%  │ ⚠ DUMB ZONE   │
╰───────┴──────────┴────────┴────────────┴────────┴───────────────╯

╭─────────────────────────────────────────────────────────────────╮
│                      Token Distribution                         │
├──────────────┬────────────┬───────────────────────────────────────┤
│ Category     │ Tokens     │ Percentage                          │
├──────────────┼────────────┼───────────────────────────────────────┤
│ system       │ 150        │ ██ 0.2%                             │
│ user         │ 12,450     │ ██████████████████ 18.1%            │
│ assistant    │ 48,200     │ ██████████████████████████████ 70.2%│
│ tools        │ 7,840      │ ███████████ 11.4%                   │
├──────────────┼────────────┼───────────────────────────────────────┤
│ TOTAL        │ 68,640     │ 40.9% of context window             │
╰──────────────┴────────────┴───────────────────────────────────────╯

╭─────────────────────────────────────────────────────────────────╮
│  ⚠  DUMB ZONE REACHED at Round 49 (40.6%)                       │
│                                                                 │
│  Recommendation: Consider summarizing or pruning context        │
│  before continuing the conversation.                            │
╰─────────────────────────────────────────────────────────────────╯
```

### Verbose Output (with `-v`)

Adds detailed breakdown within each category:

```
╭─────────────────────────────────────────────────────────────────╮
│                    Detailed Token Breakdown                     │
├──────────────────────────────────────────────────────────────────┤
│ TOOLS BREAKDOWN:                                                │
│   ├─ Tool Definitions:     2,100 tokens (26.8%)                 │
│   ├─ Tool Calls:           1,840 tokens (23.5%)                 │
│   └─ Tool Responses:       3,900 tokens (49.7%)                 │
│                                                                 │
│ USER BREAKDOWN:                                                 │
│   ├─ Text Content:        11,200 tokens (89.9%)                 │
│   └─ Images (estimated):   1,250 tokens (10.1%)                 │
╰──────────────────────────────────────────────────────────────────╯
```

## Programmatic API

### Analyzer Class

```python
from tinyloop.ctx import CTXAnalyzer, CTXResult, CTXStatus

# Initialize analyzer
analyzer = CTXAnalyzer(
    model="anthropic/claude-sonnet-4-20250514",
    context_window=168000,
    threshold=0.4,
    offline=False
)

# Full analysis of a conversation
result: CTXResult = analyzer.analyze(messages)

# Access full results
print(result.total_tokens)           # 68640
print(result.threshold_tokens)       # 67200
print(result.is_in_dumb_zone)        # True
print(result.dumb_zone_round)        # 49
print(result.percentage_used)        # 0.409
print(result.categories)             # {'system': 150, 'user': 12450, ...}
print(result.rounds)                 # List of RoundResult objects

# Simple status check (faster, less detail)
status: CTXStatus = analyzer.status(messages)
print(status.total_tokens)           # 68640
print(status.threshold_tokens)       # 67200
print(status.is_in_dumb_zone)        # True
print(status.tokens_until_dumb_zone) # -1440 (negative = over threshold)
print(status.message)                # "⚠ 68,640 / 67,200 tokens (40.9%) — IN DUMB ZONE"
```

### Point-in-Time Analysis from LLM Instance

```python
from tinyloop import LLM
from tinyloop.ctx import CTXAnalyzer

# Create LLM and have a conversation
llm = LLM(model="anthropic/claude-sonnet-4-20250514")
llm(prompt="Hello!")
llm(prompt="Tell me about Python")
llm(prompt="What about JavaScript?")

# Analyze current conversation state at any point
analyzer = CTXAnalyzer(model=llm.model, context_window=168000)

# Get the current message history and analyze
history = llm.get_history()
result = analyzer.analyze(history)

# Or get simple status
status = analyzer.status(history)
print(status.message)
# Output: ✓ 1,234 / 67,200 tokens (0.7%) — 65,966 tokens until dumb zone
```

### CTXResult Schema (Full Analysis)

```python
from pydantic import BaseModel
from typing import List, Dict, Optional

class RoundResult(BaseModel):
    round_number: int
    role: str
    tokens: int
    cumulative_tokens: int
    percentage_used: float
    is_in_dumb_zone: bool

class CTXResult(BaseModel):
    model: Optional[str]
    context_window: int
    threshold: float
    threshold_tokens: int
    total_tokens: int
    percentage_used: float
    is_in_dumb_zone: bool
    dumb_zone_round: Optional[int]  # First round that entered dumb zone
    categories: Dict[str, int]      # Tokens per category
    rounds: List[RoundResult]
    tokenizer_used: str             # 'anthropic_api', 'tiktoken', 'fallback'
```

### CTXStatus Schema (Simple Check)

```python
class CTXStatus(BaseModel):
    total_tokens: int
    threshold_tokens: int
    context_window: int
    percentage_used: float
    is_in_dumb_zone: bool
    is_approaching: bool            # Within 10% of threshold
    tokens_until_dumb_zone: int     # Positive = safe, negative = over
    message: str                    # Pre-formatted status message
```

### LLM Middleware Integration

```python
from tinyloop import LLM
from tinyloop.ctx import CTXMiddleware

# Create middleware
ctx = CTXMiddleware(
    context_window=168000,
    threshold=0.4,
    action="warn"  # or "raise"
)

# Attach to LLM instance
llm = LLM(
    model="anthropic/claude-sonnet-4-20250514",
    temperature=0.7
)
llm.add_middleware(ctx)

# Use normally - warnings will be printed when dumb zone is reached
response = llm(prompt="Hello!")

# Check status at any point during the conversation
status = ctx.get_status()
print(status.message)

if status.is_in_dumb_zone:
    # Handle context management
    pass

# Get full analysis at any point
result = ctx.get_analysis()
```

### Middleware Actions

| Action | Behavior |
|--------|----------|
| `warn` | Prints warning to stderr, continues execution |
| `raise` | Raises `CTXThresholdExceeded` exception |

```python
from tinyloop.ctx import CTXThresholdExceeded

try:
    response = llm(prompt="...")
except CTXThresholdExceeded as e:
    print(f"Dumb zone reached at {e.percentage_used:.1%}")
    print(f"Total tokens: {e.total_tokens}")
    # Implement context pruning strategy
```

## Tokenizer Implementation

### Tokenizer Selection Logic

```python
def get_tokenizer(model: str, offline: bool = False) -> Tokenizer:
    """
    Select appropriate tokenizer based on model name.

    Priority:
    1. Anthropic API (for claude-3+ models, if not offline)
    2. Tiktoken with model-specific encoding (for OpenAI models)
    3. Tiktoken with cl100k_base fallback
    """
    if not offline and is_anthropic_model(model) and is_claude_3_or_later(model):
        return AnthropicAPITokenizer(model)

    if is_openai_model(model):
        return TiktokenTokenizer(model)

    # Fallback
    return TiktokenTokenizer(encoding="cl100k_base")
```

### Model Detection

```python
def is_anthropic_model(model: str) -> bool:
    """Check if model is from Anthropic."""
    model_lower = model.lower()
    return any(x in model_lower for x in ['claude', 'anthropic'])

def is_claude_3_or_later(model: str) -> bool:
    """Check if model is Claude 3 or later (has API token counting)."""
    # Claude 3 models: claude-3-opus, claude-3-sonnet, claude-3-haiku
    # Claude 3.5 models: claude-3-5-sonnet, claude-3-5-haiku
    # Claude 4 models: claude-sonnet-4, claude-opus-4
    return any(x in model.lower() for x in [
        'claude-3', 'claude-4', 'claude-sonnet-4', 'claude-opus-4'
    ])

def is_openai_model(model: str) -> bool:
    """Check if model is from OpenAI."""
    model_lower = model.lower()
    return any(x in model_lower for x in ['gpt', 'openai', 'o1', 'o3'])
```

### Image Token Estimation

For multi-modal messages containing images, use provider-specific estimation:

```python
def estimate_image_tokens(image: dict, model: str) -> int:
    """
    Estimate tokens for an image based on provider rules.

    OpenAI: Based on image dimensions and detail level
    Anthropic: ~1,600 tokens per image (varies by size)
    Fallback: 1,000 tokens per image
    """
    if is_openai_model(model):
        return openai_image_token_estimate(image)
    elif is_anthropic_model(model):
        return anthropic_image_token_estimate(image)
    else:
        return 1000  # Conservative fallback
```

## Error Handling

### Error Types

```python
class CTXError(Exception):
    """Base exception for CTX errors."""
    pass

class CTXThresholdExceeded(CTXError):
    """Raised when dumb zone threshold is exceeded (action='raise')."""
    def __init__(self, total_tokens: int, threshold_tokens: int, percentage_used: float):
        self.total_tokens = total_tokens
        self.threshold_tokens = threshold_tokens
        self.percentage_used = percentage_used
        super().__init__(
            f"Dumb zone threshold exceeded: {percentage_used:.1%} "
            f"({total_tokens:,} / {threshold_tokens:,} tokens)"
        )

class TokenizerError(CTXError):
    """Raised when tokenization fails."""
    pass

class InvalidInputError(CTXError):
    """Raised when input format is invalid."""
    pass
```

### CLI Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success, not in dumb zone |
| 1 | Success, in dumb zone (useful for CI/CD checks) |
| 2 | Input error (invalid JSON, missing file) |
| 3 | Tokenizer error (API failure, etc.) |

## Dependencies

### Required

```toml
[project.dependencies]
typer = ">=0.9.0"
rich = ">=13.0.0"
tiktoken = ">=0.5.0"
```

### Optional

```toml
[project.optional-dependencies]
anthropic = ["anthropic>=0.18.0"]  # For accurate Claude token counting
```

## Future Considerations

### Potential Enhancements

1. **Caching**: Cache token counts for repeated analysis of the same conversation
2. **Streaming Analysis**: Analyze conversations as they grow in real-time
3. **Export Formats**: Support CSV, HTML report export
4. **Threshold Recommendations**: Suggest optimal thresholds based on model
5. **Context Pruning Suggestions**: Recommend which messages to prune
6. **Integration with Observability**: Export metrics to Langfuse/MLflow

### Out of Scope (v1.0)

- Automatic context pruning/summarization
- Token budget management
- Cost estimation (separate feature)
- Multi-conversation analysis

## References

- [Anthropic Token Counting API](https://platform.claude.com/docs/en/build-with-claude/token-counting)
- [tiktoken on PyPI](https://pypi.org/project/tiktoken/)
- [tokencost on PyPI](https://pypi.org/project/tokencost/)
- [LiteLLM Documentation](https://docs.litellm.ai/)

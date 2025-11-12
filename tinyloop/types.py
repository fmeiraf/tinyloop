from typing import Any, Dict, List, Optional

from litellm.types.utils import ModelResponse
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    function_name: str
    args: dict[str, Any]
    id: str


class ToolCallDelta(BaseModel):
    id: Optional[str] = None
    function_name: Optional[str] = None
    function_arguments: Optional[str] = None


class ToolCallResponse(BaseModel):
    content: str
    cost: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    response: Any
    raw_response: Optional[ModelResponse] = None
    tool_calls: Optional[List[ToolCall]] = None
    message_history: Optional[List[Dict[str, Any]]] = None
    cost: float
    hidden_fields: dict[str, Any]


class LLMStreamingResponse(BaseModel):
    id: str
    response: Any
    chunk: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

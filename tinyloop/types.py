from typing import Any, List, Optional

from litellm.types.utils import ModelResponse
from pydantic import BaseModel


class ToolCall(BaseModel):
    function_name: str
    args: dict[str, Any]
    tool_call_id: str


class LLMResponse(BaseModel):
    response: Any
    raw_response: ModelResponse
    tool_calls: Optional[List[ToolCall]] = None
    cost: float
    hidden_fields: dict[str, Any]

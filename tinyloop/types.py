from typing import Any

from litellm.types.utils import ModelResponse
from pydantic import BaseModel


class LLMResponse(BaseModel):
    response: Any
    raw_response: ModelResponse
    cost: float
    hidden_fields: dict[str, Any]

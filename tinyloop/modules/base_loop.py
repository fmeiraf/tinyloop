from abc import abstractmethod
from typing import List, Optional

import mlflow
from pydantic import BaseModel

from tinyloop.features.function_calling import Tool
from tinyloop.inference.litellm import LLM, ToolCall

mlflow.config.enable_async_logging(True)


class BaseLoop:
    def __init__(
        self,
        model: str,
        tools: List[Tool],
        output_format: Optional[BaseModel] = None,
        temperature: float = 1.0,
        system_prompt: str = None,
        llm_kwargs: dict = {},
    ):
        self.model = model
        self.temperature = temperature
        self.llm = LLM(
            model=self.model,
            temperature=self.temperature,
            system_prompt=system_prompt,
            **llm_kwargs,
        )
        self.output_format = output_format
        self.tools = tools
        self.tools_map = {tool.name: tool for tool in tools}

    @abstractmethod
    def __call__(self, prompt: str, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    async def acall(self, prompt: str, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def _format_tool_response(self, tool_call: ToolCall, function_response: str):
        return {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": tool_call.function_name,
            "content": function_response,
        }

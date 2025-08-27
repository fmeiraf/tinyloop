import json
import logging
from typing import Any, Dict, List, Optional

import litellm
import mlflow
from litellm.types.utils import ModelResponse
from pydantic import BaseModel

from tinyloop.features.function_calling import Tool
from tinyloop.features.vision import Image
from tinyloop.inference.base import BaseInferenceModel
from tinyloop.types import LLMResponse, ToolCall
from tinyloop.utils.mlflow import mlflow_trace

logger = logging.getLogger(__name__)

mlflow.config.enable_async_logging(True)


class LLM(BaseInferenceModel):
    """
    LLM inference model using litellm.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        use_cache: bool = False,
        system_prompt: Optional[str] = None,
        message_history: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize the inference model.

        Args:
            model: Model name or path
            temperature: Temperature for sampling
            use_cache: Whether to use_cache the model
        """
        super().__init__(
            model=model,
            temperature=temperature,
            use_cache=use_cache,
            system_prompt=system_prompt,
            message_history=message_history,
        )

        self.sync_client = litellm.completion
        self.async_client = litellm.acompletion
        self.run_cost = []

    @mlflow.trace(span_type=mlflow.entities.SpanType.LLM)
    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs,
    ) -> LLMResponse:
        return self.invoke(prompt=prompt, messages=messages, stream=stream, **kwargs)

    @mlflow_trace(mlflow.entities.SpanType.LLM)
    async def acall(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs,
    ) -> LLMResponse:
        final_response = await self.ainvoke(
            prompt=prompt, messages=messages, stream=stream, **kwargs
        )
        return final_response

    @mlflow_trace(mlflow.entities.SpanType.LLM)
    def invoke(
        self,
        prompt: Optional[str] = None,
        images: Optional[List[Image]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Tool]] = None,
        stream: bool = False,
        **kwargs,
    ) -> LLMResponse:
        if messages is None:
            messages = self.message_history
            if not prompt:
                raise ValueError("Prompt is required when messages is None")
            messages.append(self._prepare_user_message(prompt, images))

        raw_response = self.sync_client(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            caching=self.use_cache,
            stream=stream,
            tools=[tool.definition for tool in tools] if tools else None,
            **kwargs,
        )

        if raw_response.choices:
            content = raw_response.choices[0].message.content
            response = (
                self._parse_structured_output(
                    content,
                    kwargs.get("response_format"),
                )
                if kwargs.get("response_format")
                else content
            )
            cost = raw_response._hidden_params["response_cost"]
            hidden_fields = raw_response._hidden_params

            tool_calls = self._parse_tool_calls(raw_response)
            if tool_calls:
                # Add a well-formed assistant message that contains tool_calls
                # OpenAI expects `content` to be a string (use empty string when using tool_calls)
                self.add_message(
                    {
                        "role": "assistant",
                        "content": response or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function_name,
                                    "arguments": json.dumps(tc.args),
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )

            if content:
                self.add_message(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )
        else:
            response = None
            cost = 0
            hidden_fields = {}
            tool_calls = None

        self.run_cost.append(cost)

        return LLMResponse(
            response=response,
            raw_response=raw_response,
            cost=cost,
            hidden_fields=hidden_fields,
            tool_calls=tool_calls,
            message_history=self.get_history(),
        )

    async def ainvoke(
        self,
        prompt: Optional[str] = None,
        images: Optional[List[Image]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Tool]] = None,
        stream: bool = False,
        **kwargs,
    ) -> LLMResponse:
        if messages is None:
            messages = self.message_history
            if not prompt:
                raise ValueError("Prompt is required when messages is None")
            messages.append(self._prepare_user_message(prompt, images))

        raw_response = await self.async_client(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            caching=self.use_cache,
            stream=stream,
            tools=[tool.definition for tool in tools] if tools else None,
            **kwargs,
        )

        if raw_response.choices:
            content = raw_response.choices[0].message.content
            response = (
                self._parse_structured_output(
                    content,
                    kwargs.get("response_format"),
                )
                if kwargs.get("response_format")
                else content
            )
            cost = raw_response._hidden_params["response_cost"]
            hidden_fields = raw_response._hidden_params

            tool_calls = self._parse_tool_calls(raw_response)
            if tool_calls:
                # Add a well-formed assistant message that contains tool_calls
                # OpenAI expects `content` to be a string (use empty string when using tool_calls)
                self.add_message(
                    {
                        "role": "assistant",
                        "content": response or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function_name,
                                    "arguments": json.dumps(tc.args),
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )

            if content and not tool_calls:
                self.add_message(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )
        else:
            response = None
            cost = 0
            hidden_fields = {}
            tool_calls = None

        self.run_cost.append(cost)

        return LLMResponse(
            response=response,
            raw_response=raw_response,
            cost=cost,
            hidden_fields=hidden_fields,
            tool_calls=tool_calls,
            message_history=self.get_history(),
        )

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the message history.
        """
        return self.message_history

    def set_history(self, history: List[Dict[str, Any]]) -> None:
        """
        Set the message history.
        """
        self.message_history = history

    def add_message(self, message: Dict[str, Any]) -> None:
        """
        Add a message to the message history.
        """
        self.message_history.append(message)

    def get_total_cost(self) -> float:
        """
        Get cost of all runs.
        """
        return sum(self.run_cost)

    def _parse_structured_output(
        self, response: str, response_format: BaseModel
    ) -> BaseModel:
        """
        Parse a structured output from a response.
        """
        return response_format.model_validate_json(response)

    def _prepare_user_message(
        self, prompt: str, images: Optional[List[Image]] = None
    ) -> List[Dict[str, Any]]:
        """
        Prepare a user message.
        """
        if images:
            return {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image.url,
                                "format": image.mime_type,
                            },
                        }
                        for image in images
                    ],
                ],
            }

        else:
            return {"role": "user", "content": prompt}

    def _prepare_assistant_message(self, content: str) -> Dict[str, Any]:
        """
        Parse an assistant message from a response.
        """
        return {
            "role": "assistant",
            "content": content,
        }

    def _parse_tool_calls(self, raw_response: ModelResponse) -> List[ToolCall]:
        """
        Parse tool calls from a response.
        """
        if not raw_response.choices[0].message.tool_calls:
            return None

        tool_calls = []
        for tool_call in raw_response.choices[0].message.tool_calls:
            if tool_call is not None:
                tool_calls.append(
                    ToolCall(
                        function_name=tool_call.function.name,
                        args=json.loads(tool_call.function.arguments),
                        id=tool_call.id,
                    )
                )

        return tool_calls

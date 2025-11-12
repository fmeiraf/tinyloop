import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional

import litellm
import mlflow
from langfuse import observe
from pydantic import BaseModel

from tinyloop.features.function_calling import Tool
from tinyloop.features.vision import Image
from tinyloop.inference.base import BaseInferenceModel
from tinyloop.types import LLMResponse, LLMStreamingResponse, ToolCall, ToolCallDelta
from tinyloop.utils.mlflow import mlflow_trace

logger = logging.getLogger(__name__)

mlflow.config.enable_async_logging(True)


class CostTracker:
    """Tracks costs from litellm callbacks by capturing stdout."""

    def __init__(self):
        self.latest_cost = 0.0
        self._original_stdout = sys.stdout
        self.cost_received_event = None
        self.is_capturing = False

    def start_cost_capture(self):
        """Start capturing cost from stdout."""
        self.cost_received_event = asyncio.Event()
        self.latest_cost = 0.0
        self.is_capturing = True

        class CostCapturingStdout:
            def __init__(self, original_stdout, cost_tracker):
                self.original_stdout = original_stdout
                self.cost_tracker = cost_tracker

            def write(self, text):
                # Write to original stdout
                self.original_stdout.write(text)
                # Extract cost if it's in the expected format
                if self.cost_tracker.is_capturing and "tloop_final_cost=" in text:
                    try:
                        cost_str = text.split("tloop_final_cost=")[1].strip()
                        self.cost_tracker.latest_cost = float(cost_str)
                        # Signal that cost has been received
                        if self.cost_tracker.cost_received_event:
                            self.cost_tracker.cost_received_event.set()
                    except (IndexError, ValueError):
                        pass
                return len(text)

            def flush(self):
                self.original_stdout.flush()

        # Set up the capturing stdout
        sys.stdout = CostCapturingStdout(self._original_stdout, self)

    def stop_cost_capture(self):
        """Stop capturing cost and restore original stdout."""
        self.is_capturing = False
        sys.stdout = self._original_stdout

    async def wait_for_cost(self, timeout=2.0):
        """Wait for cost to be captured with a timeout."""
        if self.cost_received_event:
            try:
                await asyncio.wait_for(self.cost_received_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Cost capture timed out after {timeout}s, using 0.0")

    def get_latest_cost(self):
        """Get the latest captured cost."""
        return self.latest_cost


# Global cost tracker instance
cost_tracker = CostTracker()


async def track_cost_callback(kwargs, completion_response, start_time, end_time):
    cost = kwargs["response_cost"]
    print(f"tloop_final_cost={cost:.6f}")


litellm.success_callback = [track_cost_callback]


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

    @observe(name="litellm.completion", as_type="generation")
    @mlflow.trace(span_type=mlflow.entities.SpanType.LLM)
    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs,
    ) -> LLMResponse:
        return self.invoke(prompt=prompt, messages=messages, stream=stream, **kwargs)

    @observe(name="litellm.completion", as_type="generation")
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

    def invoke(
        self,
        prompt: Optional[str] = None,
        images: Optional[List[Image]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Tool]] = None,
        stream: bool = False,
        **kwargs,
    ) -> LLMResponse:
        if stream:
            raise ValueError("Stream is not supported for sync mode")
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
            cost = raw_response._hidden_params["response_cost"] or 0
            hidden_fields = raw_response._hidden_params

            tool_calls = self._parse_tool_calls(
                raw_response.choices[0].message.tool_calls
            )
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

        if stream:
            return self._parse_streaming_response(raw_response)

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
            cost = raw_response._hidden_params["response_cost"] or 0
            hidden_fields = raw_response._hidden_params

            tool_calls = self._parse_tool_calls(
                raw_response.choices[0].message.tool_calls
            )

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

    def _parse_tool_calls(self, raw_tool_calls: List) -> List[ToolCall]:
        """
        Parse tool calls from a response.
        """
        if not raw_tool_calls:
            return None

        tool_calls = []
        for tool_call in raw_tool_calls:
            print(f"tool_call: {tool_call}")
            if tool_call is not None:
                tool_calls.append(
                    ToolCall(
                        function_name=tool_call.function.name,
                        args=json.loads(tool_call.function.arguments),
                        id=tool_call.id,
                    )
                )

        return tool_calls

    async def _parse_streaming_response(self, stream_response) -> List[Dict[str, Any]]:
        id = None
        response = ""
        tool_call_deltas = []  # store last values for all tool calls (id, function_name, function_arguments)
        latest_tool_calls = []

        # Start cost tracking
        cost_tracker.start_cost_capture()

        async for chunk in stream_response:
            id = chunk.id if chunk.id else id

            choice_content = (
                chunk.choices[0].delta.content
                if chunk.choices[0].delta.content
                else None
            )
            # print(f"chunk: {chunk}")
            # print(f"choice_content: {choice_content}")
            if choice_content:
                # model text response
                response += choice_content or ""

            # parsing tool calls
            if not chunk.choices[0].delta.tool_calls:
                yield LLMStreamingResponse(
                    id=id,
                    response=response,
                    chunk=choice_content,
                    tool_calls=latest_tool_calls,
                )
                continue

            for i, tool_call_delta in enumerate(chunk.choices[0].delta.tool_calls):
                if tool_call_delta:
                    if i >= len(tool_call_deltas):
                        tool_call_deltas.append(
                            ToolCallDelta(
                                id=None, function_name=None, function_arguments=""
                            )
                        )
                    tool_call_deltas[i].id = (
                        tool_call_delta.id or tool_call_deltas[i].id
                    )
                    tool_call_deltas[i].function_name = (
                        tool_call_delta.function.name
                        or tool_call_deltas[i].function_name
                    )
                    tool_call_deltas[i].function_arguments += (
                        tool_call_delta.function.arguments or ""
                    )

                    try:
                        args = (
                            json.loads(tool_call_deltas[i].function_arguments)
                            if tool_call_deltas[i].function_arguments
                            else {}
                        )
                    except json.decoder.JSONDecodeError:
                        logger.warning(
                            f"Failed to parse tool call arguments: {tool_call_deltas[i].function_arguments}"
                        )
                        args = {}

                    # Create the new tool call
                    new_tool_call = ToolCall(
                        function_name=tool_call_deltas[i].function_name,
                        args=args,
                        id=tool_call_deltas[i].id,
                    )

                    # Check if a tool call with the same ID already exists
                    existing_index = None
                    for idx, existing_tool_call in enumerate(latest_tool_calls):
                        if existing_tool_call.id == new_tool_call.id:
                            existing_index = idx
                            break

                    # Replace existing or append new
                    if existing_index is not None:
                        latest_tool_calls[existing_index] = new_tool_call
                    else:
                        latest_tool_calls.append(new_tool_call)
                    yield LLMStreamingResponse(
                        id=id,
                        response=response,
                        chunk=choice_content,
                        tool_calls=latest_tool_calls,
                    )

        # adding tool calls and response to history
        if latest_tool_calls:
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
                        for tc in latest_tool_calls
                    ],
                }
            )

        if response and not latest_tool_calls:
            self.add_message(
                {
                    "role": "assistant",
                    "content": response,
                }
            )

        # Wait for cost callback to complete (with timeout)
        await cost_tracker.wait_for_cost(timeout=2.0)

        # Stop cost tracking and restore stdout
        cost_tracker.stop_cost_capture()

        # Get captured cost
        captured_cost = cost_tracker.get_latest_cost()
        print(f"captured_cost: {captured_cost}")

        # Add cost to run_cost tracking
        self.run_cost.append(captured_cost)

        yield LLMResponse(
            response=response,
            tool_calls=latest_tool_calls,
            message_history=self.get_history(),
            cost=captured_cost,
            hidden_fields={},
        )

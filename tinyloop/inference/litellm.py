import logging
from typing import Any, Dict, List, Optional

import litellm
from pydantic import BaseModel

from tinyloop.functionality.vision import Image
from tinyloop.inference.base import BaseInferenceModel
from tinyloop.types import LLMResponse

logger = logging.getLogger(__name__)


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

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs,
    ) -> LLMResponse:
        return self.invoke(prompt=prompt, messages=messages, stream=stream, **kwargs)

    async def acall(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs,
    ) -> LLMResponse:
        return await self.ainvoke(
            prompt=prompt, messages=messages, stream=stream, **kwargs
        )

    def invoke(
        self,
        prompt: Optional[str] = None,
        images: Optional[List[Image]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs,
    ) -> LLMResponse:
        if messages is None:
            messages = self.message_history
            if not prompt:
                raise ValueError("Prompt is required when messages is None")
            messages.append(self._prepate_user_message(prompt, images))

        raw_response = self.sync_client(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            caching=self.use_cache,
            stream=stream,
            **kwargs,
        )

        if raw_response.choices:
            response = (
                self._parse_structured_output(
                    raw_response.choices[0].message.content,
                    kwargs.get("response_format"),
                )
                if kwargs.get("response_format")
                else raw_response.choices[0].message.content
            )
            cost = raw_response._hidden_params["response_cost"]
            hidden_fields = raw_response._hidden_params
            self.add_message(
                {
                    "role": "assistant",
                    "content": raw_response.choices[0].message.content,
                }
            )
        else:
            response = None
            cost = 0
            hidden_fields = {}

        self.run_cost.append(cost)

        return LLMResponse(
            response=response,
            raw_response=raw_response,
            cost=cost,
            hidden_fields=hidden_fields,
        )

    async def ainvoke(
        self,
        prompt: Optional[str] = None,
        images: Optional[List[Image]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs,
    ) -> LLMResponse:
        if messages is None:
            messages = self.message_history
            if not prompt:
                raise ValueError("Prompt is required when messages is None")
            messages.append(self._prepate_user_message(prompt, images))

        raw_response = await self.async_client(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            caching=self.use_cache,
            stream=stream,
            **kwargs,
        )

        if raw_response.choices:
            response = (
                self._parse_structured_output(
                    raw_response.choices[0].message.content,
                    kwargs.get("response_format"),
                )
                if kwargs.get("response_format")
                else raw_response.choices[0].message.content
            )
            cost = raw_response._hidden_params["response_cost"]
            hidden_fields = raw_response._hidden_params
            self.add_message(
                {
                    "role": "assistant",
                    "content": raw_response.choices[0].message.content,
                }
            )
        else:
            response = None
            cost = 0
            hidden_fields = {}

        self.run_cost.append(cost)

        return LLMResponse(
            response=response,
            raw_response=raw_response,
            cost=cost,
            hidden_fields=hidden_fields,
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

    def _prepate_user_message(
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

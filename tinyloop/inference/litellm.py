"""
Base LLM inference model.
"""

import logging
from typing import Any, Dict, List, Optional

import litellm

from tinyloop.inference.base import BaseInferenceModel

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

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs,
    ) -> str:
        return self.invoke(prompt=prompt, messages=messages, stream=stream, **kwargs)

    async def acall(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs,
    ) -> str:
        return await self.ainvoke(
            prompt=prompt, messages=messages, stream=stream, **kwargs
        )

    def invoke(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs,
    ) -> str:
        if messages is None:
            messages = self.message_history
            if not prompt:
                raise ValueError("Prompt is required when messages is None")
            messages.append({"role": "user", "content": prompt})

        response = self.sync_client(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            caching=self.use_cache,
            stream=stream,
            **kwargs,
        )
        return response

    async def ainvoke(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs,
    ) -> str:
        if messages is None:
            messages = self.message_history
            if not prompt:
                raise ValueError("Prompt is required when messages is None")
            messages.append({"role": "user", "content": prompt})

        response = await self.async_client(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            caching=self.use_cache,
            stream=stream,
            **kwargs,
        )
        return response

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

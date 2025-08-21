"""
Base LLM inference model.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BaseInferenceModel(ABC):
    """
    Abstract base class for inference models.

    This class defines the interface that all inference models must implement.
    It provides common functionality for model loading, inference, and cleanup.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        use_cache: bool = False,
        use_instructor: bool = True,
        system_prompt: Optional[str] = None,
        message_history: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize the inference model.

        Args:
            model: Model name or path
            temperature: Temperature for sampling
            cache: Whether to cache the model
        """
        self.model = model
        self.temperature = temperature
        self.use_cache = use_cache
        self.message_history = message_history or []
        self.use_instructor = use_instructor
        if system_prompt:
            self.message_history.append({"role": "system", "content": system_prompt})

    @abstractmethod
    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> str:
        """
        Generate a response from the model.

        This method must be implemented by subclasses to generate a response
        from the model.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    async def acall(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> str:
        """
        Generate a response from the model.

        This method must be implemented by subclasses to generate a response
        from the model.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def invoke(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> str:
        """
        Generate a response from the model.

        This method must be implemented by subclasses to generate a response
        from the model.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    async def ainvoke(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> str:
        """
        Generate a response from the model.

        This method must be implemented by subclasses to generate a response
        from the model.
        """
        raise NotImplementedError("Subclasses must implement this method")

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

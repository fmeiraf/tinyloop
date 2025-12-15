import logging
import os
from typing import List

import mlflow
from pydantic import BaseModel

from tinyloop.features.function_calling import Tool
from tinyloop.modules.base_loop import BaseLoop
from tinyloop.utils.observability import set_trace_custom

logger = logging.getLogger(__name__)


def _truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _maybe_enable_mlflow_litellm_autolog() -> None:
    """
    Opt-in MLflow LiteLLM autologging.

    Import-time autologging has surprising side effects (e.g. creating `mlruns/`).
    To enable, set `TINYLOOP_ENABLE_MLFLOW=1` (or pass enable_mlflow=True to ToolLoop).
    """

    if not _truthy_env(os.getenv("TINYLOOP_ENABLE_MLFLOW")):
        return
    try:
        mlflow.litellm.autolog()
    except Exception:
        logger.exception("Failed to enable MLflow LiteLLM autologging")


class ToolLoop(BaseLoop):
    def __init__(
        self,
        model: str,
        tools: List[Tool],
        output_format: BaseModel,
        max_iterations: int = 5,
        temperature: float = 1.0,
        system_prompt: str = None,
        llm_kwargs: dict = {},
        enable_mlflow: bool | None = None,
    ):
        if enable_mlflow is True:
            os.environ["TINYLOOP_ENABLE_MLFLOW"] = "1"
        elif enable_mlflow is False:
            os.environ["TINYLOOP_ENABLE_MLFLOW"] = "0"
        _maybe_enable_mlflow_litellm_autolog()

        def finish_func():
            return True

        tools.append(
            Tool(
                name="finish",
                description="Use this tool when you are done and want to finish the task",
                func=finish_func,
            )
        )

        super().__init__(
            model=model,
            tools=tools,
            output_format=output_format,
            temperature=temperature,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
        )
        self.max_iterations = max_iterations

    @set_trace_custom(
        mlflow.entities.SpanType.AGENT, lambda self, func: "tinyloop.tool_loop"
    )
    def __call__(self, prompt: str, **kwargs):
        self.llm.add_message(self.llm._prepare_user_message(prompt))
        for _ in range(self.max_iterations):
            response = self.llm(
                messages=self.llm.get_history(), tools=self.tools, **kwargs
            )
            if response.tool_calls:
                should_finish = False
                for tool_call in response.tool_calls:
                    tool_response = self.tools_map[tool_call.function_name](
                        **tool_call.args
                    )

                    self.llm.add_message(
                        self._format_tool_response(tool_call, str(tool_response))
                    )

                    if tool_call.function_name == "finish":
                        should_finish = True
                        break

                if should_finish:
                    break
        final_response = self.llm(
            messages=self.llm.get_history(),
            response_format=self.output_format,
        )
        return final_response

    @set_trace_custom(
        mlflow.entities.SpanType.AGENT, lambda self, func: "tinyloop.tool_loop"
    )
    async def acall(self, prompt: str, **kwargs):
        self.llm.add_message(self.llm._prepare_user_message(prompt))
        for _ in range(self.max_iterations):
            response = await self.llm.acall(
                messages=self.llm.get_history(), tools=self.tools, **kwargs
            )
            if response.tool_calls:
                should_finish = False
                for tool_call in response.tool_calls:
                    tool_response = await self.tools_map[tool_call.function_name].acall(
                        **tool_call.args
                    )

                    self.llm.add_message(
                        self._format_tool_response(tool_call, str(tool_response))
                    )

                    if tool_call.function_name == "finish":
                        should_finish = True
                        break

                if should_finish:
                    break

        final_response = await self.llm.acall(
            messages=self.llm.get_history(), response_format=self.output_format
        )
        return final_response

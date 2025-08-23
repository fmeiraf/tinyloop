from typing import List

import mlflow
from pydantic import BaseModel

from tinyloop.features.function_calling import Tool
from tinyloop.modules.base_loop import BaseLoop


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
    ):
        tools.append(
            Tool(
                name="finish",
                description="Use this tool when you are done and want to finish the loop",
                func=lambda: True,
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

    @mlflow.trace(span_type=mlflow.entities.SpanType.AGENT)
    def __call__(self, prompt: str, **kwargs):
        messages = [self.llm._prepare_user_message(prompt)]
        for _ in range(self.max_iterations):
            response = self.llm(messages=messages, tools=self.tools, **kwargs)
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    if tool_call.function_name == "finish":
                        break

                    tool_response = self.tools_map[tool_call.function_name](
                        **tool_call.args
                    )
                    print(tool_response)
                    self.llm.add_message(
                        self._format_tool_response(tool_call, str(tool_response))
                    )
                    print(self.llm.get_history())

        return self.llm(
            messages=messages,
            response_format=self.output_format,
        )

    async def acall(self, prompt: str, **kwargs):
        pass

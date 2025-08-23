from pydantic import BaseModel

from tinyloop.inference.litellm import LLM


class Generate:
    def __init__(
        self,
        model: str,
        output_format: BaseModel,
        temperature: float = 1.0,
        system_prompt: str = None,
        **kwargs,
    ):
        self.model = model
        self.temperature = temperature
        self.llm = LLM(
            model=self.model,
            temperature=self.temperature,
            system_prompt=system_prompt,
            **kwargs,
        )
        self.output_format = output_format

    def __call__(self, prompt: str, **kwargs):
        return self.llm(prompt, response_format=self.output_format, **kwargs)

    async def acall(self, prompt: str, **kwargs):
        return await self.llm.acall(
            prompt, response_format=self.output_format, **kwargs
        )

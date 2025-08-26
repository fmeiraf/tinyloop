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

    def call(self, prompt: str):
        return self.llm(prompt)

    async def acall(self, prompt: str):
        result = await self.llm.acall(prompt)
        return result

    @classmethod
    def run(
        cls,
        prompt: str,
        model: str,
        output_format: BaseModel,
        temperature: float = 1.0,
        system_prompt: str = None,
        **kwargs,
    ):
        """Initialize and call the Generate class in a single step."""
        instance = cls(
            model=model,
            output_format=output_format,
            temperature=temperature,
            system_prompt=system_prompt,
            **kwargs,
        )
        return instance.call(prompt)

    @classmethod
    async def arun(
        cls,
        prompt: str,
        model: str,
        output_format: BaseModel,
        temperature: float = 1.0,
        system_prompt: str = None,
        **kwargs,
    ):
        """Initialize and call the Generate class asynchronously in a single step."""
        instance = cls(
            model=model,
            output_format=output_format,
            temperature=temperature,
            system_prompt=system_prompt,
            **kwargs,
        )
        result = await instance.acall(prompt)
        return result

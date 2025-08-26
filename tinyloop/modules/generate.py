from tinyloop.inference.litellm import LLM


class Generate:
    def __init__(
        self,
        model: str,
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

    def call(self, prompt: str, **kwargs):
        return self.llm(prompt, **kwargs)

    async def acall(self, prompt: str, **kwargs):
        result = await self.llm.acall(prompt, **kwargs)
        return result

    @classmethod
    def run(
        cls,
        prompt: str,
        model: str,
        temperature: float = 1.0,
        system_prompt: str = None,
        llm_kwargs: dict = {},
        **kwargs,
    ):
        """Initialize and call the Generate class in a single step."""
        instance = cls(
            model=model,
            temperature=temperature,
            system_prompt=system_prompt,
        )
        return instance.call(prompt, **kwargs)

    @classmethod
    async def arun(
        cls,
        prompt: str,
        model: str,
        temperature: float = 1.0,
        system_prompt: str = None,
        llm_kwargs: dict = {},
        **kwargs,
    ):
        """Initialize and call the Generate class asynchronously in a single step."""
        instance = cls(
            model=model,
            temperature=temperature,
            system_prompt=system_prompt,
        )
        result = await instance.acall(prompt, **kwargs)
        return result

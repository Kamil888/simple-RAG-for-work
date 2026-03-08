import openai
from .base import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, model: str = "gpt-4o"):
        self.client = openai.OpenAI()
        self.model = model

    def complete(self, system_prompt: str, user_message: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return resp.choices[0].message.content

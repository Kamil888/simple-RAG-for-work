import anthropic
from .base import BaseLLM


class ClaudeLLM(BaseLLM):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic()
        self.model = model

    def complete(self, system_prompt: str, user_message: str) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return msg.content[0].text

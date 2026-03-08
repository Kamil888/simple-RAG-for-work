from abc import ABC, abstractmethod


class BaseLLM(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_message: str) -> str:
        ...

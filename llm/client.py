from typing import Dict, Type

from llm.base_llm_client import BaseLLMClient
from llm.gigachat import GigaChatLLMClient
from llm.mistral import MistralLLMClient
from llm.openai import OpenAILLMClient


class LLMFactory:
    """
    Factory for creating LLM clients.
    """

    _clients: Dict[str, Type[BaseLLMClient]] = {
        "openai": OpenAILLMClient,
        "mistral": MistralLLMClient,
        "gigachat": GigaChatLLMClient,
    }

    @classmethod
    def create_llm(cls, config) -> BaseLLMClient:
        model_type = config.model_type.lower()
        client_class = cls._clients.get(model_type)
        if not client_class:
            raise ValueError(f"Unsupported LLM type: {model_type}")
        return client_class(config)

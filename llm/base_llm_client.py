from abc import ABC, abstractmethod

from langchain_core.language_models import BaseLLM

from configs import LLMConfig


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def get_langchain_llm(self) -> BaseLLM:
        """Get LangChain LLM instance."""
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate connection to the LLM service."""
        pass

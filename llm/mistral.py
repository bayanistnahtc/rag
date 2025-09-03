import logging

from langchain_core.language_models import BaseLLM
from langchain_mistralai import ChatMistralAI

from llm.base_llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class MistralLLMClient(BaseLLMClient):
    """Mistral LLM client implementation"""

    def get_langchain_llm(self) -> BaseLLM:
        """Get Mistral LangChain LLM instance."""
        try:
            return ChatMistralAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                mistral_api_key=self.config.model_api_key,
                # request_timeout=self.config.timeout
            )
        except Exception as e:
            logger.error(f"Failed to create Mistral LLM client: {e}")
            raise

    def validate_connection(self) -> bool:
        """Validate Mistral connection."""
        try:
            llm = self.get_langchain_llm()
            # Simple test invocation
            llm.invoke("Test connection")
            return True
        except Exception as e:
            logger.error(f"MistralLLM connection validation failed: {e}")
            return False

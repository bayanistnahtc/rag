import logging

from langchain_core.language_models import BaseLLM
from langchain_openai import ChatOpenAI

from llm.base_llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class OpenAILLMClient(BaseLLMClient):
    """OpenAI LLM client implementation."""

    def get_langchain_llm(self) -> BaseLLM:
        """Get OpenAI LangChain LLM instance."""
        try:
            return ChatOpenAI(
                model_name=self.config.model_name,
                openai_api_key=self.config.model_api_key,
                openai_api_base=self.config.model_url,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                request_timeout=self.config.timeout,
            )
        except Exception as e:
            logger.error(f"Failed to create OpenAI LLM client: {e}")
            raise

    def validate_connection(self) -> bool:
        """Validate OpenAI connection."""
        try:
            llm = self.get_langchain_llm()
            # Simple test invocation
            llm.invoke("Test connection")
            return True
        except Exception as e:
            logger.error(f"OpenAI connection validation failed: {e}")
            return False

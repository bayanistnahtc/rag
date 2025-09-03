import logging

from langchain_community.chat_models import GigaChat
from langchain_core.language_models import BaseLLM

from llm.base_llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class GigaChatLLMClient(BaseLLMClient):
    """GigaChat LLM client implementation."""

    def get_langchain_llm(self) -> BaseLLM:
        """Get GigaChat LangChain LLM instance."""
        try:
            return GigaChat(
                credentials=self.config.model_api_key,
                scope="GIGACHAT_API_PERS",
                verify_ssl_certs=False,
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
        except Exception as e:
            logger.error(f"Failed to create GigaChat LLM client: {e}")
            raise

    def validate_connection(self) -> bool:
        """
        Validate GigaChat connection.
        """
        try:
            llm = self.get_langchain_llm()
            llm.invoke("Test connection")
            return True
        except Exception as e:
            logger.error(f"GigaChat connection validation failed: {e}")
            return False

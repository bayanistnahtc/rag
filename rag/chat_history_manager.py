import logging
from typing import Any, Dict, List, Optional

from langchain.memory import (
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory,
    ConversationTokenBufferMemory,
)
from langchain.memory.chat_memory import BaseChatMemory
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import BaseMessage

from configs.rag_configs import ChatHistoryConfig

logger = logging.getLogger(__name__)


class HistoryManager:
    """Manage chat history with context size control"""

    def __init__(self, config: ChatHistoryConfig):
        self.config = config
        self._memory_instances: Dict[str, Any] = {}

    def create_memory_for_session(
        self, session_id: str, llm: Optional[BaseLanguageModel] = None
    ) -> BaseChatMemory:
        """
        Create memory instance for session from configs
        """
        if session_id in self._memory_instances:
            return self._memory_instances[session_id]

        memory_type = self.config.memory_type.lower()

        if memory_type == "token_buffer":
            if llm is None:
                logger.warning(
                    "LLM not provided for token_buffer memory, "
                    "using fallback window memory"
                )
                memory = ConversationBufferWindowMemory(
                    k=self.config.window_size, return_messages=True
                )
            else:
                memory = ConversationTokenBufferMemory(
                    llm=llm,
                    max_token_limit=self.config.max_token_limit,
                    return_messages=True,
                )
        elif memory_type == "window":
            memory = ConversationBufferWindowMemory(
                k=self.config.window_size, return_messages=True
            )
        elif memory_type == "summary_buffer":
            if llm is None:
                logger.warning(
                    "LLM not provided for summary_buffer memory, "
                    "using fallback window memory"
                )
                memory = ConversationBufferWindowMemory(
                    k=self.config.window_size, return_messages=True
                )
            else:
                memory = ConversationSummaryBufferMemory(
                    llm=llm,
                    max_token_limit=self.config.max_token_limit,
                    return_messages=True,
                )
        else:
            raise ValueError(f"Unsupported memory type: {memory_type}")

        self._memory_instances[session_id] = memory
        logger.info(f"Created {memory_type} memory for session {session_id}")
        return memory

    def get_chat_history_for_session(self, session_id: str) -> BaseChatMessageHistory:
        """ """
        if session_id not in self._memory_instances:
            return InMemoryChatMessageHistory()

        memory = self._memory_instances[session_id]

        if hasattr(memory, "chat_memory"):
            return memory.chat_memory
        else:
            return InMemoryChatMessageHistory()

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """ """
        if session_id not in self._memory_instances:
            return {
                "exist": False,
                "memory_type": None,
                "message_count": 0,
                "esimated_tokens": 0,
            }

        chat_history = self.get_chat_history_for_session(session_id)
        messages = chat_history.messages if hasattr(chat_history, "messages") else []

        return {
            "exists": True,
            "memory_type": self.config.memory_type,
            "message_count": len(messages),
            "estimated_tokens": self._estimate_tokens(messages),
            "max_token_limit": self.config.max_token_limit,
            "window_size": self.config.window_size,
        }

    def clear_session(self, session_id: str) -> bool:
        """
        Crear session history
        """
        if session_id not in self._memory_instances:
            return False

        memory = self._memory_instances[session_id]
        if hasattr(memory, "clear"):
            memory.clear()
        elif hasattr(memory, "chat_memory") and hasattr(memory.chat_memory, "clear"):
            memory.chat_memory.clear()

        logger.info(f"Cleared session {session_id}")
        return True

    def cleanup_session(self, session_id: str) -> bool:
        """
        Remove session from memory instances
        """
        if session_id in self._memory_instances:
            del self._memory_instances[session_id]
            logger.info(f"Cleaned up session {session_id}")
            return True
        return False

    def get_active_sessions(self) -> List[str]:
        """
        Get list of active session IDs
        """
        return list(self._memory_instances.keys())

    def _estimate_tokens(self, messages: List[BaseMessage]) -> int:
        """
        Estimate token count for messges
        """
        if not messages:
            return 0

        total_content = ""
        for msg in messages:
            if hasattr(msg, "content") and msg.content:
                total_content += str(msg.content) + " "

        # It is assumed that in avg num of tokens is bigger tha num of words
        word_count = len(total_content.split())
        return int(word_count * 1.3)

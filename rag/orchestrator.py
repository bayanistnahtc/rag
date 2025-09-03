import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
)

from configs import RAGConfig, RecursiveCharacterSplitterConfig, SemanticSplitterConfig
from configs.app_settings import Settings
from data_processing.loader_factory import DataLoaderFactory
from llm.client import LLMFactory
from rag.chat_history_manager import HistoryManager
from rag.models import Answer
from rag.prompt_templates import (
    ANSWER_GENERATION_PROMPT,
    FALLBACK_CLARIFICATION_WITH_QUESTION,
    FALLBACK_CLARIFICATION_WITHOUT_QUESTION,
    create_clarification_prompt,
)
from rag.utils import docs_to_answer, format_documents
from vector_store.factory import VectorStoreFactory
from vector_store.vector_store import FAISSBMExtractRetriever

logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """The orchestrator of the main rag pipeline"""

    def __init__(self, config: RAGConfig, settings: Optional[Settings] = None):
        self.config = config
        self.settings = settings or Settings()
        self.data_loader = None
        self.vector_store: Optional[FAISSBMExtractRetriever] = None
        self.retriever = None
        self.compressor = None
        self.llm = None
        self._history_manager: Optional[HistoryManager] = None
        self._rag_chain = None

    def initialize(self, force_rebuild_storage: bool = False) -> None:
        """
        Initialize all RAG components

        Args:
            force_rebuild_storage: Whether to force rebuild of indexes

        Raises:
            Exception: If initialization failed
        """

        try:
            logger.info("Initializing RAG pipeline...")

            # Initialize data loader
            self._initialize_data_loader()

            # Initialize vector store
            self.vector_store = VectorStoreFactory.create_faiss_bm25_store(
                embedding_config=self.config.embedding,
                vector_config=self.config.vector_store,
                compression_config=self.config.compression,
            )

            # Load or create indexes
            if self.vector_store is None:
                raise ValueError(
                    "'vector_store' not initialized. Check initialization first."
                )

            if force_rebuild_storage or not self.vector_store.index_exists():
                logger.info("Creating new indexes...")
                self._build_indexes()
            else:
                logger.info("Loading existing indexes...")
                self.vector_store.load_index()

            # Initialize retriever
            self.retriever = self.vector_store.get_extended_retriever(
                self.config.retrieval, self.config.compression
            )

            # Initialize LLM
            llm_client = LLMFactory.create_llm(self.config.llm)
            self.llm = llm_client.get_langchain_llm()

            # Initialize history manager
            self._history_manager = HistoryManager(config=self.config.chat_history)

            # Build RAG chain
            self._build_chain()
            logger.info("RAG pipeline initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise

    def _build_indexes(self, equipment: Optional[str] = None) -> None:
        """
        Build vector indexes from data

        Raises:
            ValueError: If no documents loaded
            Exception: If index building fails
        """
        try:
            # Load documents - always load all documents for MVP
            logger.info("Loading documents...")
            if self.data_loader is None:
                raise ValueError(
                    "'data_loader' not initialized. Check initialization first."
                )

            # Always load all documents, equipment filtering happens during retrieval
            documents = self.data_loader.load_documents(use_local=False)

            if not documents:
                raise ValueError("No documents loaded")

            # Process documents
            logger.info("Processing documents...")
            processed_documents = self.data_loader.process_documents(documents)

            # Create indexes
            logger.info("Creating vector indexes...")
            self.vector_store.create_index(processed_documents)

            # Save indexes
            logger.info("Saving indexes to disk...")
            self.vector_store.save_index()

            logger.info(
                (
                    f"Successfully build indexes from "
                    f"{len(processed_documents)} document chunks"
                )
            )

        except Exception as e:
            logger.error(f"Failed to build indexes: {e}")
            raise

    def _initialize_data_loader(self) -> None:
        """
        Initialize data loader component.
        """
        try:
            chunking_config = cast(
                Union[SemanticSplitterConfig, RecursiveCharacterSplitterConfig],
                self.config.text_splitter,
            )

            self.data_loader = DataLoaderFactory.create_documents_loader(
                documents_config=self.config.documents_config,
                chunking_config=chunking_config,
                vector_store_config=self.config.vector_store,
                settings=self.settings,
            )
        except Exception as e:
            logger.error(f"Failed to initialize data loader: {e}")
            raise

    def _convert_chat_history_to_messages(
        self, chat_history: List[Tuple[str, str]]
    ) -> List[BaseMessage]:
        """
        Convert chat history tuples to LangChain message objects.

        Args:
            chat_history: List of (human_message, ai_message) tuples

        Returns:
            List of BaseMessage objects
        """
        messages = []
        for human_msg, ai_msg in chat_history:
            messages.append(HumanMessage(content=human_msg))
            messages.append(AIMessage(content=ai_msg))
        return messages

    def _build_chain(self):
        """
        Build RAG chain using LangChain v0.3.x best practices for conversational RAG.
        Uses proper message history handling without clarification logic.

        Raises:
            RuntimeError: If chain building fails
            ValueError: If required components are missing
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized")
        if not self.llm:
            raise ValueError("LLM not initialized")

        try:
            # Function to contextualize question based on chat history
            def contextualize_question(inputs):
                question = inputs["question"]
                chat_history = inputs.get("chat_history", [])

                # If no chat history, return original question
                if not chat_history:
                    return question

                # Convert chat history to message format for better context
                messages = self._convert_chat_history_to_messages(chat_history)

                # Create contextualization prompt
                history_text = chr(10).join(
                    [
                        f"{'Пользователь' if isinstance(msg, HumanMessage)
                           else 'Ассистент'}: "
                        f"{msg.content}"
                        for msg in messages[-6:]
                    ]
                )

                contextualize_prompt = f"""
                Учитывая историю диалога и новый вопрос пользователя,
                переформулируй вопрос так,
                чтобы он был понятен без контекста предыдущих сообщений.

                История диалога:
                {history_text}

                Новый вопрос: {question}

                Переформулированный вопрос:"""

                try:
                    contextualized = self.llm.invoke(contextualize_prompt)
                    result = (
                        contextualized.content
                        if hasattr(contextualized, "content")
                        else str(contextualized)
                    )
                    return result.strip()
                except Exception as e:
                    logger.warning(f"Failed to contextualize question: {e}")
                    return question

            # Use hybrid_retrieve for retrieval, now with equipment support
            def hybrid_retrieve_with_equipment(inputs):
                query = inputs["contextualized_question"]
                equipment = inputs.get("equipment")
                return self.vector_store.hybrid_retrieve(
                    query, self.retriever, equipment=equipment
                )

            self._rag_chain = (
                RunnablePassthrough.assign(
                    contextualized_question=RunnableLambda(contextualize_question)
                )
                | RunnablePassthrough.assign(
                    retrieved_docs=RunnableLambda(hybrid_retrieve_with_equipment)
                )
                | RunnablePassthrough.assign(
                    context=lambda x: format_documents(x["retrieved_docs"])
                )
                | RunnablePassthrough.assign(
                    answer_text=(
                        RunnableLambda(
                            lambda x: {
                                "question": x["contextualized_question"],
                                "context": x["context"],
                            }
                        )
                        | ANSWER_GENERATION_PROMPT
                        | self.llm
                        | StrOutputParser()
                    )
                )
                | RunnableLambda(
                    lambda x: {
                        "answer_text": x["answer_text"],
                        "retrieved_docs": x["retrieved_docs"],
                        "response_type": (
                            "ANSWER"
                            if len(x["retrieved_docs"]) > 0
                            else "CLARIFICATION"
                        ),
                    }
                )
                | RunnableLambda(docs_to_answer)
            )
            logger.info("Conversational RAG chain built successfully")
        except Exception as e:
            logger.error(f"Failed to build RAG chain: {e}")
            raise

    def invoke(
        self,
        query: str,
        session_id: Optional[str] = None,
        equipment: Optional[str] = None,
    ) -> Answer:
        """
        Invoke the RAG pipeline with a query and session-based chat history.

        Args:
            query: User's question
            session_id: Optional session identifier for chat history management
            equipment: Optional equipment identifier for filtering

        Returns:
            Answer object with generated response and source documents

        Raises:
            ValueError: If RAG pipeline is not initialized
        """
        if not self._rag_chain:
            raise ValueError("RAG pipeline not initialized. Call initialize() first.")

        if not self._history_manager:
            raise ValueError(
                "History manager not initialized. Call initialize() first."
            )

        # Get chat history from session
        chat_history = []
        if session_id:
            memory = self._history_manager.create_memory_for_session(
                session_id, self.llm
            )
            chat_history_messages = (
                memory.chat_memory.messages if hasattr(memory, "chat_memory") else []
            )
            # Convert messages to tuples for backward compatibility
            for i in range(0, len(chat_history_messages), 2):
                if i + 1 < len(chat_history_messages):
                    human_msg = chat_history_messages[i]
                    ai_msg = chat_history_messages[i + 1]
                    if hasattr(human_msg, "content") and hasattr(ai_msg, "content"):
                        chat_history.append((human_msg.content, ai_msg.content))

        # Pass all context as input to the chain
        result = self._rag_chain.invoke(
            {"question": query, "chat_history": chat_history, "equipment": equipment}
        )

        # Add the new interaction to history if session_id provided
        if session_id and result.answer_text:
            memory = self._history_manager.create_memory_for_session(
                session_id, self.llm
            )
            memory.chat_memory.add_user_message(query)
            memory.chat_memory.add_ai_message(result.answer_text)

        return result

    def clear_session_history(self, session_id: str) -> bool:
        """
        Clear chat history for a specific session.

        Args:
            session_id: Session identifier

        Returns:
            bool: True if session was cleared, False if session not found
        """
        if not self._history_manager:
            raise ValueError(
                "History manager not initialized. Call initialize() first."
            )

        return self._history_manager.clear_session(session_id)

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific session.

        Args:
            session_id: Session identifier

        Returns:
            Dict with session statistics
        """
        if not self._history_manager:
            raise ValueError(
                "History manager not initialized. Call initialize() first."
            )

        return self._history_manager.get_session_stats(session_id)

    def get_active_sessions(self) -> List[str]:
        """
        Get list of all active session IDs.

        Returns:
            List of active session identifiers
        """
        if not self._history_manager:
            raise ValueError(
                "History manager not initialized. Call initialize() first."
            )

        return self._history_manager.get_active_sessions()

    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions based on TTL configuration.

        Returns:
            Number of sessions cleaned up
        """
        if not self._history_manager:
            raise ValueError(
                "History manager not initialized. Call initialize() first."
            )

        # For now, implement basic cleanup - in production you'd check timestamps
        active_sessions = self._history_manager.get_active_sessions()
        cleaned_count = 0

        # Simple cleanup: remove sessions that haven't been accessed recently
        # In a real implementation, you'd track last access time
        for session_id in active_sessions:
            stats = self._history_manager.get_session_stats(session_id)
            if stats.get("message_count", 0) == 0:
                self._history_manager.cleanup_session(session_id)
                cleaned_count += 1

        return cleaned_count

    def generate_clarification_question(
        self, question: str = "", session_id: Optional[str] = None
    ) -> str:
        """
        Generate a clarification question based on the user's query and session history.

        Args:
            question: The user's original question (optional, can be empty)
            session_id: Optional session identifier to get context from history

        Returns:
            str: Generated clarification question

        Raises:
            ValueError: If LLM is not initialized or
                if both question and session_id are empty
        """
        if not self.llm:
            raise ValueError("LLM not initialized. Call initialize() first.")

        if not question.strip() and not session_id:
            raise ValueError("Either question or session_id must be provided")

        # Get context from session history if available
        context = ""
        if session_id and self._history_manager:
            memory = self._history_manager.create_memory_for_session(
                session_id, self.llm
            )
            chat_history_messages = (
                memory.chat_memory.messages if hasattr(memory, "chat_memory") else []
            )

            # Get last few messages for context
            recent_messages = chat_history_messages[-4:]  # Last 2 exchanges
            if recent_messages:
                context_parts = []
                for msg in recent_messages:
                    role = (
                        "Пользователь" if isinstance(msg, HumanMessage) else "Ассистент"
                    )
                    context_parts.append(f"{role}: {msg.content}")
                context = "\n".join(context_parts)

        try:
            # Create clarification prompt with question and context
            clarification_prompt = create_clarification_prompt(question, context)

            # For prompts without question, we don't need to pass question parameter
            clarification_input = {}
            if question.strip():
                clarification_input["question"] = question

            # Generate clarification question
            result = clarification_prompt.invoke(clarification_input)
            response = self.llm.invoke(result)

            clarification_text = (
                response.content if hasattr(response, "content") else str(response)
            ).strip()

            logger.info(f"Generated clarification question for session {session_id}")
            return clarification_text

        except Exception as e:
            logger.error(f"Failed to generate clarification question: {e}")
            # Fallback clarification question
            if question.strip():
                return FALLBACK_CLARIFICATION_WITH_QUESTION.format(question=question)
            else:
                return FALLBACK_CLARIFICATION_WITHOUT_QUESTION

    async def health_check(self) -> Dict[str, Dict[str, Any]]:
        """
        Perform comprehensive health check of all RAG components.

        Returns: Dict[str, Dict[str, Any]]
            Dict mapping component names to their health status
        """
        components = {}

        # Check vector store
        try:
            if self.vector_store and self.vector_store.index_exists():
                components["vector_store"] = {
                    "status": "healthy",
                    "index_size": self.vector_store.get_index_size(),
                }
            else:
                components["vector_store"] = {
                    "status": "unhealthy",
                    "error": "Index not found",
                }
        except Exception as e:
            components["vector_store"] = {"status": "unhealthy", "error": str(e)}

        # Check LLM
        try:
            if self.llm:
                # Quick health check with simple query
                start_time = time.time()
                test_response = self.llm.invoke("Test")
                latency = (time.time() - start_time) * 1000
                components["llm"] = {
                    "status": "healthy" if test_response else "degraded",
                    "latency_ms": round(latency, 2),
                }
            else:
                components["llm"] = {
                    "status": "unhealthy",
                    "error": "LLM not initialized",
                }
        except Exception as e:
            components["llm"] = {"status": "unhealthy", "error": str(e)}

        # Check retriever
        try:
            if self.retriever:
                # Test retrieval with simple query
                test_docs = self.retriever.invoke("test", k=1)
                components["retriever"] = {
                    "status": "healthy",
                    "test_results": len(test_docs),
                }
            else:
                components["retriever"] = {
                    "status": "unhealthy",
                    "error": "Retriever not initialized",
                }
        except Exception as e:
            components["retriever"] = {"status": "unhealthy", "error": str(e)}

        # Check history manager
        try:
            if self._history_manager:
                active_sessions = len(self._history_manager.get_active_sessions())
                components["history_manager"] = {
                    "status": "healthy",
                    "active_sessions": active_sessions,
                }
            else:
                components["history_manager"] = {
                    "status": "unhealthy",
                    "error": "History manager not initialized",
                }
        except Exception as e:
            components["history_manager"] = {"status": "unhealthy", "error": str(e)}

        return components

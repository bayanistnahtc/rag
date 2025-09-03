import logging
import pickle
from typing import Optional

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from configs import (
    CompressionConfig,
    EmbeddingConfig,
    RetrievalConfig,
    VectorStoreConfig,
)
from rag.reranker import get_compressor_retriever

logger = logging.getLogger(__name__)


class FAISSBMExtractRetriever:
    """
    Hybrid retriever combining FAISS and BM25.
    """

    DEFAULT_FAISS_INDEX_NAME = "faiss_index"
    DEFAULT_BM25_INDEX_NAME = "bm25_retriever.pkl"
    DEFAULT_DOCUMENTS_INDEX_NAME = "documents.pkl"

    def __init__(
        self,
        embedding_config: EmbeddingConfig,
        vector_config: VectorStoreConfig,
        compression_config: Optional[CompressionConfig] = None,
    ):
        self.embedding_config = embedding_config
        self.vector_config = vector_config
        self.compression_config = compression_config
        self.embeddings = None
        self.faiss_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self._documents = None

        # Initialize embeddings
        self._initialize_embeddings()

    def create_index(self, documents):
        """
        Create vector index from documents
        """
        logger.info(f"Creating vector index from {len(documents)} documents...")

        if not documents:
            raise ValueError("Cannot create index from empty document list")

        self._documents = documents

        # Create FAISS index
        logger.info("Creating FAISS index...")
        self.faiss_store = FAISS.from_documents(
            documents=documents, embedding=self.embeddings
        )

        # Create BM25 retriever
        logger.info("Creating BM25 retriever...")
        # NOTE: BM25 parameters are left at default.
        # TODO: Make parameters k1 and b configurable via config.
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        logger.info("Vector index created successfully")

    def save_index(self) -> None:
        """
        Save vector index to disk
        """
        if not self.faiss_store or not self.bm25_retriever:
            raise ValueError("No index to save. Create index first.")

        # Ensure directory exists
        self.vector_config.index_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss_path = self.vector_config.index_path / "faiss_index"
        logger.info(f"Saving FAISS index to {faiss_path}")
        self.faiss_store.save_local(str(faiss_path))

        # Save BM25 retriever
        bm25_path = self.vector_config.index_path / "bm25_retriever.pkl"
        logger.info(f"Saving BM25 retriever to {bm25_path}")
        with open(bm25_path, "wb") as f:
            # Security note: We control the serialization, files are stored locally
            pickle.dump(self.bm25_retriever, f)  # nosec B301

        # Save documents for reference
        docs_path = self.vector_config.index_path / "documents.pkl"
        logger.info(f"Saving documents to {docs_path}")
        with open(docs_path, "wb") as f:
            # Security note: We control the serialization, files are stored locally
            pickle.dump(self._documents, f)  # nosec B301

        logger.info("Vector index saved successfully")

    def index_exists(self) -> bool:
        """
        Check if indexes exist on disk
        """
        faiss_path = self.vector_config.index_path / self.DEFAULT_FAISS_INDEX_NAME
        bm25_path = self.vector_config.index_path / self.DEFAULT_BM25_INDEX_NAME
        docs_path = self.vector_config.index_path / self.DEFAULT_DOCUMENTS_INDEX_NAME

        return faiss_path.exists() and bm25_path.exists() and docs_path.exists()

    def load_index(self) -> None:
        """
        Load vector index from disk
        """
        try:
            index_path = self.vector_config.index_path
            if not index_path.exists():
                raise FileNotFoundError(f"Index path {index_path} does not exists")

            faiss_path = self.vector_config.index_path / self.DEFAULT_FAISS_INDEX_NAME
            bm25_path = self.vector_config.index_path / self.DEFAULT_BM25_INDEX_NAME
            docs_path = (
                self.vector_config.index_path / self.DEFAULT_DOCUMENTS_INDEX_NAME
            )

            # Load FAISS index
            logger.info(f"Loading FAISS index from {faiss_path}")
            self.faiss_store = FAISS.load_local(
                faiss_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )

            # Load BM25 retriever
            logger.info(f"Loading BM25 retriever from {bm25_path}")
            with open(bm25_path, "rb") as f:
                # Security note: Files are locally stored and trusted
                self.bm25_retriever = pickle.load(f)  # nosec B301

            # Load documents if they exist
            if docs_path.exists():
                logger.info(f"Loading documents from {docs_path}")
                with open(docs_path, "rb") as f:
                    # Security note: Files are locally stored and trusted
                    self._documents = pickle.load(f)  # nosec B301

            logger.info("Vector index loaded successfully")
        except Exception as e:
            logger.info(f"Failed to load indexes: {e}")

    def get_retriever(self, retrieval_config: RetrievalConfig) -> BaseRetriever:
        """
        Get ensemble retriever
        """
        if not self.faiss_store or not self.bm25_retriever:
            raise ValueError("Index not loaded. Load or create index first.")

        if self.ensemble_retriever is None:
            # Configure retrievers
            faiss_retriever = self.faiss_store.as_retriever(
                search_kwargs={"k": retrieval_config.k}
            )

            self.bm25_retriever.k = retrieval_config.k * 10

            # Create ensemble retriever
            # NOTE: The weights for the ensemble are currently hardcoded.
            # This is definitely a candidate for being moved to the config for tuning.
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[faiss_retriever, self.bm25_retriever],
                weights=[retrieval_config.faiss_weight, retrieval_config.bm25_weight],
            )

        return self.ensemble_retriever

    def get_extended_retriever(
        self, retrieval_config: RetrievalConfig, compression_config: CompressionConfig
    ):
        """
        Get extended retriever with compression.

        Args:
            retrieval_config (RetrievalConfig): The retrieval configuration.
            compression_config (CompressionConfig): The compression configuration.

        Returns:
            BaseRetriever: The extended retriever with compression.
        """
        # Create a retriever with compression
        base_retriever = self.get_retriever(retrieval_config)
        if not base_retriever:
            raise ValueError("Index not loaded. Load or create index first.")

        if not compression_config:
            raise ValueError("CompressionConfig is required for extended retriever.")

        compression_retriever = get_compressor_retriever(
            base_retriever, compression_config
        )

        return compression_retriever

    def get_index_size(self) -> int:
        """
        Return the number of documents currently indexed.

        Returns: int
            Number of indexed documents.
        """
        if self._documents is None:
            raise ValueError("'_documents' is None. Check initialization first.")
        return len(self._documents) if self._documents else 0

    def substring_retrieve(self, query, k=3):
        """
        Fallback: Return up to k chunks containing the query substring
        (case-insensitive).
        """
        if not self._documents:
            return []
        matches = []
        q = query.lower().strip()
        for doc in self._documents:
            if q and q in doc.page_content.lower():
                matches.append(doc)
            if len(matches) >= k:
                break
        return matches

    def hybrid_retrieve(
        self,
        query,
        compression_retriever,
        equipment: Optional[str] = None,
    ):
        """
        Hybrid retrieve with optional equipment filtering.
        Use ensemble (FAISS+BM25), then supplement with reranking and compression.

        Args:
            query (str): The search query.
            compression_retriever (BaseRetriever): The compression retriever to use.
            equipment (Optional[str]): Optional equipment filter.

        Returns:
            List[Document]: The list of retrieved documents.
        """

        results = compression_retriever.invoke(
            query, search_kwargs={"filter": {"file_name": equipment}}
        )

        # Filter by equipment (file_name) if specified
        if equipment:
            filtered_results = []
            for doc in results:
                doc_source = doc.metadata.get("file_name", "").lower()
                if equipment.lower() in doc_source or doc_source in equipment.lower():
                    filtered_results.append(doc)
            results = filtered_results[:20]
        return results

    def _initialize_embeddings(self) -> None:
        """
        Initialize embedding model
        """
        logger.info(f"Initialize embeddings: {self.embedding_config.model_name}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_config.model_name,
            model_kwargs=self.embedding_config.model_kwargs,
            encode_kwargs=self.embedding_config.encode_kwargs,
        )

        logger.info("Embeddings initialized successfully")

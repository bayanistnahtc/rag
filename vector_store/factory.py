from typing import Optional

from configs.rag_configs import CompressionConfig, EmbeddingConfig, VectorStoreConfig
from vector_store.vector_store import FAISSBMExtractRetriever


class VectorStoreFactory:
    """Factory for creating vector stores"""

    @staticmethod
    def create_faiss_bm25_store(
        embedding_config: EmbeddingConfig,
        vector_config: VectorStoreConfig,
        compression_config: Optional[CompressionConfig] = None,
    ) -> FAISSBMExtractRetriever:
        """
        Create FAISS + BM25 hybrid vector store
        """
        return FAISSBMExtractRetriever(
            embedding_config, vector_config, compression_config
        )

import logging

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_huggingface import HuggingFaceEmbeddings

from configs.rag_configs import CompressionConfig

logger = logging.getLogger(__name__)


def get_compressor_retriever(retriever, config: CompressionConfig):
    """Create a ContextualCompressionRetriever configured from env.

    Args:
        retriever: Base retriever to wrap.
        config (CompressionConfig): Settings for embeddings and filter.
    """

    logger.info(
        "Initializing compression retriever with model '%s', k=%s, threshold=%.2f",
        config.model_name,
        config.top_k,
        config.similarity_threshold,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=config.model_name,
        model_kwargs=config.model_kwargs,
        encode_kwargs=config.encode_kwargs,
        query_encode_kwargs=config.query_encode_kwargs,
    )

    compressor = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=config.similarity_threshold,
        k=config.top_k,
    )

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever,
    )

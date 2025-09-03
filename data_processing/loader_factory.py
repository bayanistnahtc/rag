import logging
from typing import Union

from configs import (
    DocumentsConfig,
    RecursiveCharacterSplitterConfig,
    SemanticSplitterConfig,
    VectorStoreConfig,
)
from configs.app_settings import DocumentSource, Settings
from configs.rag_configs import PageBasedSplitterConfig, ParagraphSplitterConfig
from data_processing.loader import PDFLoader
from data_processing.s3_loader import S3PDFLoader

logger = logging.getLogger(__name__)


class DataLoaderFactory:
    """
    Factory for creating document loaders by file type and source.
    """

    _loaders = {
        ".pdf": PDFLoader,
    }

    @staticmethod
    def create_documents_loader(
        documents_config: DocumentsConfig,
        chunking_config: Union[
            SemanticSplitterConfig,
            RecursiveCharacterSplitterConfig,
            PageBasedSplitterConfig,
            ParagraphSplitterConfig,
        ],
        vector_store_config: VectorStoreConfig,
        settings: Settings,
    ):
        """
        Create appropriate document loader based on document source configuration.

        Args:
            documents_config: Configuration for document processing
            chunking_config: Configuration for text chunking
            vector_store_config: Configuration for vector store
            settings: Application settings

        Returns:
            Appropriate document loader instance
        """
        if settings.document_source == DocumentSource.S3:
            logger.info("Creating S3 PDF loader")
            return S3PDFLoader(
                documents_config=documents_config,
                vector_store_config=vector_store_config,
                chunking_config=chunking_config,
                settings=settings,
            )
        else:
            logger.info("Creating local PDF loader")
            return PDFLoader(
                documents_config=documents_config,
                vector_store_config=vector_store_config,
                chunking_config=chunking_config,
            )

    # When calling load_documents, pass equipment as needed
    # loader.load_documents(equipment=...)

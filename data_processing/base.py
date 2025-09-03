"""Base classes for data processing components."""

from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class BaseLoader(ABC):
    """Abstract interface for a document loader."""

    @abstractmethod
    def load_documents(self, use_local: bool = False) -> List[Document]:
        """
        Loads documents from a given source.
        The source can be a file path, URL, API endpoint, etc.

        Args:
            use_local (bool): Whether to load from local cache

        Raises:
            Exception: If loading fails
        """
        pass

    @abstractmethod
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process a list of documents."""
        pass


class BaseTextSplitter(ABC):
    """Abstract interface for a text splitter."""

    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of documents into smaller chunks.

        Args:
            documents List[Document]: A list of documents to be split.

        Return (List[Document]): A list of of chunked documents.
        """
        pass

import logging
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

from configs import DocumentsConfig, VectorStoreConfig
from data_processing.base import BaseLoader
from data_processing.splitters import TextSplitterFactory

logger = logging.getLogger(__name__)


class PDFLoader(BaseLoader):
    """
    A downloader for PDF documents using PyMuPDF.
    Provides basic text extraction.
    """

    def __init__(
        self,
        documents_config: DocumentsConfig,
        chunking_config,
        vector_store_config: VectorStoreConfig,
    ):
        self.documents_config = documents_config
        self.chunking_config = chunking_config
        self.vector_store_config = vector_store_config
        self._text_splitter = None
        self.save_cache = False

    @property
    def text_splitter(self):
        """
        Initialization of text splitter

        Returns:
            Configured text splitter instance
        """
        if self._text_splitter is None:
            try:
                self._text_splitter = TextSplitterFactory.create(self.chunking_config)
            except Exception as e:
                logger.error(f"Failed to initialize text_splitter: {e}")
                raise
        return self._text_splitter

    def load_documents(
        self, use_local: bool = False, equipment: Optional[str] = None
    ) -> List[Document]:
        """
        Load all PDF documents under a directory (or single file).
        If use_local is True, pull from cache instead.
        """
        try:
            # NOTE: PyMuPDFLoader handles text PDFs well,
            # but scanned documents will require OCR, such as Tesseract.
            # TODO: Add OCR logic in the future, possibly via Unstructured.

            if use_local:
                documents = self._load_local_documents()
            else:
                documents = self._load_pdf_documents(equipment=equipment)
                if not documents:
                    raise ValueError(
                        f"No PDF files found at {self.documents_config.file_path!r}"
                    )
                if self.save_cache:
                    raise NotImplementedError
                    # self._cache_documents(documents)

            if not documents:
                raise ValueError("No documents were loaded")

            logger.info(f"Successfully loaded {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Failed to load PDF {documents[0]}: {e}")
            # NOTE: For now we're just throwing an exception,
            # but we need more granular error handling.
            raise

    def process_documents(self, documents):
        if not documents:
            raise ValueError("No documents provided for processing")

        try:
            all_chunks = self.text_splitter.split_documents(documents)
            logger.info(
                f"Processed {len(documents)} documents into {len(all_chunks)} chunks"
            )

            return all_chunks
        except Exception as e:
            logger.error(f"Failed to peocess documents: {e}")
            raise

    def _load_local_documents(self) -> List[Document]:
        """
        Load documents from local cache
        Returns:
            List[Document]: List of cached documents.
        """
        raise NotImplementedError("Local cache loading is not implemented")

    def _load_pdf_documents(self, equipment: Optional[str] = None) -> List[Document]:
        """
        Walk self.documents_config.file_path (file or directory),
        load all matching PDFs via PyMuPDF, annotate metadata, and return.
        Args:
            equipment (Optional[str]): Equipment identifier to filter documents by
            subdirectory.
        Returns:
            List[Document]: List of loaded documents with metadata.

        Raises:
            ValueError: If no PDF files are found at the specified path.
        """

        root = Path(self.documents_config.file_path)
        exts = {e.lower() for e in self.documents_config.file_extensions}
        all_docs: List[Document] = []

        if not root.exists():
            raise ValueError(f"Path does not exist: {root!r}")

        # Gather all .pdf files under the directory (or single file)
        if root.is_dir():
            if equipment:
                # Filter by subdirectory matching equipment name or id
                equipment_dir = next(
                    (d for d in root.iterdir() if equipment.lower() in d.name.lower()),
                    None,
                )
                if equipment_dir and equipment_dir.is_dir():
                    pdf_files = [
                        p for p in equipment_dir.rglob("*") if p.suffix.lower() in exts
                    ]
                else:
                    pdf_files = []
            else:
                pdf_files = [p for p in root.rglob("*") if p.suffix.lower() in exts]
        else:
            pdf_files = [root] if root.suffix.lower() in exts else []

        if not pdf_files:
            return []

        for pdf_path in pdf_files:
            try:
                loader = PyMuPDFLoader(file_path=str(pdf_path))
                docs = loader.load()

                # Annotate each chunk with source metadata
                for i, doc in enumerate(docs):
                    # Extract equipment information from file path
                    equipment_info = self._extract_equipment_from_path(pdf_path)

                    doc.metadata.update(
                        {
                            "file_name": pdf_path.name,
                            "file_path": str(pdf_path),
                            "file_type": "pdf",
                            "equipment": equipment_info,
                            "page_number": i + 1,  # Add page number (1-indexed)
                            "total_pages": len(docs),
                            "source": str(
                                pdf_path
                            ),  # Add source field for vector store filtering
                        }
                    )

                all_docs.extend(docs)
                logger.info(f"Loaded {len(docs)} pages from {pdf_path.name}")

            except Exception as e:
                logger.error(f"Error loading {pdf_path!r}: {e}")
                # continue with next file rather than failing everything
                continue

        return all_docs

    def _extract_equipment_from_path(self, pdf_path: Path) -> str:
        """
        Extract equipment identifier from file path.
        Assumes equipment is in the directory structure.

        Args:
            pdf_path (Path): Path to the PDF file.
        Returns:
            str: Equipment identifier, or "unknown" if not found.
        """
        try:
            # Try to find equipment in parent directories
            for parent in pdf_path.parents:
                if parent.name and parent.name != pdf_path.parent.name:
                    # Return the first meaningful directory name as equipment
                    return parent.name
            # Fallback to filename without extension
            return pdf_path.stem
        except Exception:
            return "unknown"

"""S3/MinIO document loader for PDF files."""

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from minio import Minio
from minio.error import S3Error

from configs import DocumentsConfig, VectorStoreConfig
from configs.app_settings import Settings
from data_processing.base import BaseLoader
from data_processing.splitters import TextSplitterFactory

logger = logging.getLogger(__name__)


class S3PDFLoader(BaseLoader):
    """
    A document loader for PDF files stored in S3/MinIO.
    Downloads PDFs from S3 bucket and processes them using PyMuPDF.
    """

    def __init__(
        self,
        documents_config: DocumentsConfig,
        chunking_config,
        vector_store_config: VectorStoreConfig,
        settings: Settings,
    ):
        """
        Initialize S3 PDF loader.

        Args:
            documents_config: Configuration for document processing
            chunking_config: Configuration for text chunking
            vector_store_config: Configuration for vector store
            settings: Application settings containing S3 configuration
        """
        self.documents_config = documents_config
        self.chunking_config = chunking_config
        self.vector_store_config = vector_store_config
        self.settings = settings
        self._text_splitter = None
        self._minio_client = None

    @property
    def text_splitter(self):
        """
        Initialize text splitter lazily.

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

    @property
    def minio_client(self) -> Minio:
        """
        Initialize MinIO client lazily.

        Returns:
            Configured MinIO client instance
        """
        if self._minio_client is None:
            try:
                self._minio_client = Minio(
                    endpoint=self.settings.s3_endpoint,
                    access_key=self.settings.s3_access_key,
                    secret_key=self.settings.s3_secret_key,
                    secure=self.settings.s3_secure,
                    region=self.settings.s3_region,
                )
                logger.info(f"Initialized MinIO client for endpoint: {self.settings.s3_endpoint}")
            except Exception as e:
                logger.error(f"Failed to initialize MinIO client: {e}")
                raise
        return self._minio_client

    def load_documents(
        self, use_local: bool = False, equipment: Optional[str] = None
    ) -> List[Document]:
        """
        Load PDF documents from S3 bucket.

        Args:
            use_local: Not used for S3 loader (kept for interface compatibility)
            equipment: Optional equipment identifier to filter documents

        Returns:
            List of loaded documents with metadata

        Raises:
            ValueError: If no PDF files are found or bucket doesn't exist
            S3Error: If S3 operations fail
        """
        try:
            # Ensure bucket exists
            self._ensure_bucket_exists()

            # Load documents from S3
            documents = self._load_s3_documents(equipment=equipment)
            
            if not documents:
                raise ValueError(
                    f"No PDF files found in S3 bucket '{self.settings.s3_bucket_name}'"
                )

            logger.info(f"Successfully loaded {len(documents)} documents from S3")
            return documents

        except Exception as e:
            logger.error(f"Failed to load documents from S3: {e}")
            raise

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process documents by splitting them into chunks.

        Args:
            documents: List of documents to process

        Returns:
            List of processed document chunks

        Raises:
            ValueError: If no documents provided
        """
        if not documents:
            raise ValueError("No documents provided for processing")

        try:
            all_chunks = self.text_splitter.split_documents(documents)
            logger.info(
                f"Processed {len(documents)} documents into {len(all_chunks)} chunks"
            )
            return all_chunks
        except Exception as e:
            logger.error(f"Failed to process documents: {e}")
            raise

    def _ensure_bucket_exists(self) -> None:
        """
        Ensure the S3 bucket exists, create if it doesn't.

        Raises:
            S3Error: If bucket operations fail
        """
        try:
            bucket_name = self.settings.s3_bucket_name
            
            if not self.minio_client.bucket_exists(bucket_name):
                logger.info(f"Creating S3 bucket: {bucket_name}")
                self.minio_client.make_bucket(bucket_name)
                logger.info(f"Successfully created S3 bucket: {bucket_name}")
            else:
                logger.debug(f"S3 bucket already exists: {bucket_name}")
                
        except S3Error as e:
            logger.error(f"S3 error while ensuring bucket exists: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while ensuring bucket exists: {e}")
            raise

    def _load_s3_documents(self, equipment: Optional[str] = None) -> List[Document]:
        """
        Load PDF documents from S3 bucket.

        Args:
            equipment: Optional equipment identifier to filter documents by prefix

        Returns:
            List of loaded documents with metadata
        """
        all_docs: List[Document] = []
        bucket_name = self.settings.s3_bucket_name
        
        try:
            # Set prefix for equipment filtering
            prefix = f"{equipment}/" if equipment else ""
            
            # List objects in bucket
            objects = self.minio_client.list_objects(
                bucket_name, 
                prefix=prefix, 
                recursive=True
            )
            
            pdf_objects = [
                obj for obj in objects 
                if obj.object_name.lower().endswith('.pdf')
            ]
            
            if not pdf_objects:
                logger.warning(f"No PDF files found in bucket '{bucket_name}' with prefix '{prefix}'")
                return []

            logger.info(f"Found {len(pdf_objects)} PDF files in S3 bucket")

            # Process each PDF file
            for obj in pdf_objects:
                try:
                    docs = self._process_s3_pdf(obj.object_name)
                    all_docs.extend(docs)
                    logger.info(f"Loaded {len(docs)} pages from {obj.object_name}")
                except Exception as e:
                    logger.error(f"Error processing {obj.object_name}: {e}")
                    # Continue with next file rather than failing everything
                    continue

            return all_docs

        except S3Error as e:
            logger.error(f"S3 error while listing objects: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while loading S3 documents: {e}")
            raise

    def _process_s3_pdf(self, object_name: str) -> List[Document]:
        """
        Download and process a single PDF file from S3.

        Args:
            object_name: S3 object key/name

        Returns:
            List of documents (pages) from the PDF

        Raises:
            S3Error: If S3 download fails
        """
        bucket_name = self.settings.s3_bucket_name
        temp_path = None
        
        try:
            # Create temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.pdf')
            
            try:
                # Close the file descriptor to allow other processes to access the file
                os.close(temp_fd)
                
                # Download from S3
                self.minio_client.fget_object(bucket_name, object_name, temp_path)
                
                # Load PDF using PyMuPDF
                loader = PyMuPDFLoader(file_path=temp_path)
                docs = loader.load()
                
                # Add metadata to each document
                for i, doc in enumerate(docs):
                    equipment_info = self._extract_equipment_from_s3_key(object_name)
                    
                    doc.metadata.update({
                        "file_name": Path(object_name).name,
                        "s3_key": object_name,
                        "s3_bucket": bucket_name,
                        "file_type": "pdf",
                        "equipment": equipment_info,
                        "page_number": i + 1,
                        "total_pages": len(docs),
                        "source": f"s3://{bucket_name}/{object_name}",
                        "document_source": "s3",
                    })
                
                return docs
                
            finally:
                # Clean up temporary file
                if temp_path and Path(temp_path).exists():
                    try:
                        Path(temp_path).unlink()
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup temp file {temp_path}: {cleanup_error}")
                        
        except S3Error as e:
            logger.error(f"S3 error while downloading {object_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing PDF {object_name}: {e}")
            raise

    def _extract_equipment_from_s3_key(self, s3_key: str) -> str:
        """
        Extract equipment identifier from S3 object key.
        
        Assumes equipment is in the directory structure of the S3 key.
        For example: "equipment1/manuals/device_manual.pdf" -> "equipment1"

        Args:
            s3_key: S3 object key/path

        Returns:
            Equipment identifier, or "unknown" if not found
        """
        try:
            if not s3_key:
                return "unknown"
                
            # Split the S3 key by '/' and take the first part as equipment
            parts = s3_key.split('/')
            if len(parts) > 1:
                return parts[0]
            else:
                # If no directory structure, use filename without extension
                return Path(s3_key).stem
        except Exception:
            return "unknown"

    def upload_pdf_to_s3(self, local_file_path: str, s3_key: str) -> bool:
        """
        Upload a PDF file to S3 bucket.
        
        This is a utility method for uploading PDFs to the S3 bucket.

        Args:
            local_file_path: Path to local PDF file
            s3_key: S3 object key where the file will be stored

        Returns:
            True if upload successful, False otherwise
        """
        try:
            bucket_name = self.settings.s3_bucket_name
            
            # Ensure bucket exists
            self._ensure_bucket_exists()
            
            # Upload file
            self.minio_client.fput_object(
                bucket_name, 
                s3_key, 
                local_file_path,
                content_type='application/pdf'
            )
            
            logger.info(f"Successfully uploaded {local_file_path} to s3://{bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload {local_file_path} to S3: {e}")
            return False

    def list_s3_objects(self, prefix: str = "") -> List[str]:
        """
        List all objects in the S3 bucket with optional prefix.

        Args:
            prefix: Optional prefix to filter objects

        Returns:
            List of object keys/names
        """
        try:
            bucket_name = self.settings.s3_bucket_name
            objects = self.minio_client.list_objects(
                bucket_name, 
                prefix=prefix, 
                recursive=True
            )
            return [obj.object_name for obj in objects]
        except Exception as e:
            logger.error(f"Failed to list S3 objects: {e}")
            return []
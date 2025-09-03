"""Unit tests for S3 PDF loader."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.documents import Document
from minio.error import S3Error

from configs import DocumentsConfig, VectorStoreConfig
from configs.app_settings import Settings
from data_processing.s3_loader import S3PDFLoader


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Settings()
    settings.s3_endpoint = "localhost:9000"
    settings.s3_access_key = "testkey"
    settings.s3_secret_key = "testsecret"
    settings.s3_bucket_name = "test-bucket"
    settings.s3_secure = False
    settings.s3_region = "us-east-1"
    return settings


@pytest.fixture
def documents_config():
    """Create documents config for testing."""
    return DocumentsConfig(file_path="test/path")


@pytest.fixture
def vector_store_config():
    """Create vector store config for testing."""
    return VectorStoreConfig()


@pytest.fixture
def chunking_config():
    """Create mock chunking config for testing."""
    config = Mock()
    config.type = "recursive_character"
    return config


@pytest.fixture
def s3_loader(mock_settings, documents_config, vector_store_config, chunking_config):
    """Create S3 loader instance for testing."""
    return S3PDFLoader(
        documents_config=documents_config,
        chunking_config=chunking_config,
        vector_store_config=vector_store_config,
        settings=mock_settings,
    )


class TestS3PDFLoader:
    """Test cases for S3PDFLoader class."""

    def test_init(self, s3_loader, mock_settings):
        """Test S3PDFLoader initialization."""
        assert s3_loader.settings == mock_settings
        assert s3_loader._minio_client is None
        assert s3_loader._text_splitter is None

    @patch('data_processing.s3_loader.Minio')
    def test_minio_client_property(self, mock_minio_class, s3_loader, mock_settings):
        """Test MinIO client property initialization."""
        mock_client = Mock()
        mock_minio_class.return_value = mock_client
        
        # First access should create client
        client = s3_loader.minio_client
        assert client == mock_client
        
        # Verify client was created with correct parameters
        mock_minio_class.assert_called_once_with(
            endpoint=mock_settings.s3_endpoint,
            access_key=mock_settings.s3_access_key,
            secret_key=mock_settings.s3_secret_key,
            secure=mock_settings.s3_secure,
            region=mock_settings.s3_region,
        )
        
        # Second access should return same client
        client2 = s3_loader.minio_client
        assert client2 == mock_client
        assert mock_minio_class.call_count == 1

    @patch('data_processing.s3_loader.TextSplitterFactory')
    def test_text_splitter_property(self, mock_factory, s3_loader, chunking_config):
        """Test text splitter property initialization."""
        mock_splitter = Mock()
        mock_factory.create.return_value = mock_splitter
        
        # First access should create splitter
        splitter = s3_loader.text_splitter
        assert splitter == mock_splitter
        mock_factory.create.assert_called_once_with(chunking_config)
        
        # Second access should return same splitter
        splitter2 = s3_loader.text_splitter
        assert splitter2 == mock_splitter
        assert mock_factory.create.call_count == 1

    @patch('data_processing.s3_loader.Minio')
    def test_ensure_bucket_exists_creates_bucket(self, mock_minio_class, s3_loader):
        """Test bucket creation when bucket doesn't exist."""
        mock_client = Mock()
        mock_client.bucket_exists.return_value = False
        mock_minio_class.return_value = mock_client
        
        s3_loader._ensure_bucket_exists()
        
        mock_client.bucket_exists.assert_called_once_with("test-bucket")
        mock_client.make_bucket.assert_called_once_with("test-bucket")

    @patch('data_processing.s3_loader.Minio')
    def test_ensure_bucket_exists_bucket_already_exists(self, mock_minio_class, s3_loader):
        """Test when bucket already exists."""
        mock_client = Mock()
        mock_client.bucket_exists.return_value = True
        mock_minio_class.return_value = mock_client
        
        s3_loader._ensure_bucket_exists()
        
        mock_client.bucket_exists.assert_called_once_with("test-bucket")
        mock_client.make_bucket.assert_not_called()

    @patch('data_processing.s3_loader.Minio')
    def test_ensure_bucket_exists_s3_error(self, mock_minio_class, s3_loader):
        """Test S3 error handling in bucket creation."""
        mock_client = Mock()
        mock_client.bucket_exists.side_effect = S3Error(
            "Test error", "test", "test", "test", "test", "test"
        )
        mock_minio_class.return_value = mock_client
        
        with pytest.raises(S3Error):
            s3_loader._ensure_bucket_exists()

    def test_extract_equipment_from_s3_key(self, s3_loader):
        """Test equipment extraction from S3 key."""
        # Test with directory structure
        assert s3_loader._extract_equipment_from_s3_key("equipment1/manuals/test.pdf") == "equipment1"
        
        # Test with nested structure
        assert s3_loader._extract_equipment_from_s3_key("eq1/subdir/file.pdf") == "eq1"
        
        # Test with single file (no directory)
        assert s3_loader._extract_equipment_from_s3_key("test.pdf") == "test"
        
        # Test with empty string - should return empty string, not "unknown"
        assert s3_loader._extract_equipment_from_s3_key("") == "unknown"

    @patch('data_processing.s3_loader.Minio')
    def test_list_s3_objects(self, mock_minio_class, s3_loader):
        """Test listing S3 objects."""
        mock_client = Mock()
        mock_obj1 = Mock()
        mock_obj1.object_name = "file1.pdf"
        mock_obj2 = Mock()
        mock_obj2.object_name = "file2.pdf"
        mock_client.list_objects.return_value = [mock_obj1, mock_obj2]
        mock_minio_class.return_value = mock_client
        
        objects = s3_loader.list_s3_objects("test-prefix")
        
        assert objects == ["file1.pdf", "file2.pdf"]
        mock_client.list_objects.assert_called_once_with(
            "test-bucket", prefix="test-prefix", recursive=True
        )

    @patch('data_processing.s3_loader.Minio')
    def test_list_s3_objects_error(self, mock_minio_class, s3_loader):
        """Test error handling in list S3 objects."""
        mock_client = Mock()
        mock_client.list_objects.side_effect = Exception("Test error")
        mock_minio_class.return_value = mock_client
        
        objects = s3_loader.list_s3_objects()
        assert objects == []

    @patch('data_processing.s3_loader.os')
    @patch('data_processing.s3_loader.Path')
    @patch('data_processing.s3_loader.tempfile')
    @patch('data_processing.s3_loader.PyMuPDFLoader')
    @patch('data_processing.s3_loader.Minio')
    def test_process_s3_pdf(self, mock_minio_class, mock_loader_class, mock_tempfile, mock_path, mock_os, s3_loader):
        """Test processing a single S3 PDF."""
        # Setup mocks
        mock_client = Mock()
        mock_minio_class.return_value = mock_client
        
        # Mock tempfile.mkstemp
        mock_tempfile.mkstemp.return_value = (123, "/tmp/test.pdf")  # (fd, path)
        
        mock_doc1 = Document(page_content="Page 1", metadata={})
        mock_doc2 = Document(page_content="Page 2", metadata={})
        mock_loader = Mock()
        mock_loader.load.return_value = [mock_doc1, mock_doc2]
        mock_loader_class.return_value = mock_loader
        
        mock_path_obj = Mock()
        mock_path_obj.name = "test.pdf"
        mock_path_obj.exists.return_value = True
        mock_path_obj.unlink.return_value = None
        mock_path.return_value = mock_path_obj
        
        # Test
        docs = s3_loader._process_s3_pdf("equipment1/test.pdf")
        
        # Verify
        assert len(docs) == 2
        assert docs[0].metadata["file_name"] == "test.pdf"
        assert docs[0].metadata["s3_key"] == "equipment1/test.pdf"
        assert docs[0].metadata["equipment"] == "equipment1"
        assert docs[0].metadata["page_number"] == 1
        assert docs[1].metadata["page_number"] == 2
        
        # Verify tempfile operations
        mock_tempfile.mkstemp.assert_called_once_with(suffix='.pdf')
        mock_os.close.assert_called_once_with(123)
        
        mock_client.fget_object.assert_called_once_with(
            "test-bucket", "equipment1/test.pdf", "/tmp/test.pdf"
        )

    @patch.object(S3PDFLoader, '_load_s3_documents')
    @patch.object(S3PDFLoader, '_ensure_bucket_exists')
    def test_load_documents_success(self, mock_ensure_bucket, mock_load_s3, s3_loader):
        """Test successful document loading."""
        mock_doc = Document(page_content="Test", metadata={})
        mock_load_s3.return_value = [mock_doc]
        
        docs = s3_loader.load_documents(equipment="test-equipment")
        
        assert len(docs) == 1
        assert docs[0] == mock_doc
        mock_ensure_bucket.assert_called_once()
        mock_load_s3.assert_called_once_with(equipment="test-equipment")

    @patch.object(S3PDFLoader, '_load_s3_documents')
    @patch.object(S3PDFLoader, '_ensure_bucket_exists')
    def test_load_documents_no_files(self, mock_ensure_bucket, mock_load_s3, s3_loader):
        """Test loading when no files found."""
        mock_load_s3.return_value = []
        
        with pytest.raises(ValueError, match="No PDF files found"):
            s3_loader.load_documents()

    @patch('data_processing.s3_loader.TextSplitterFactory')
    def test_process_documents_success(self, mock_factory, s3_loader):
        """Test successful document processing."""
        mock_doc = Document(page_content="Test", metadata={})
        mock_chunk1 = Document(page_content="Chunk 1", metadata={})
        mock_chunk2 = Document(page_content="Chunk 2", metadata={})
        
        mock_splitter = Mock()
        mock_splitter.split_documents.return_value = [mock_chunk1, mock_chunk2]
        mock_factory.create.return_value = mock_splitter
        
        chunks = s3_loader.process_documents([mock_doc])
        
        assert len(chunks) == 2
        assert chunks == [mock_chunk1, mock_chunk2]
        mock_splitter.split_documents.assert_called_once_with([mock_doc])

    def test_process_documents_no_documents(self, s3_loader):
        """Test processing with no documents."""
        with pytest.raises(ValueError, match="No documents provided"):
            s3_loader.process_documents([])

    @patch('data_processing.s3_loader.Minio')
    def test_upload_pdf_to_s3_success(self, mock_minio_class, s3_loader):
        """Test successful PDF upload to S3."""
        mock_client = Mock()
        mock_client.bucket_exists.return_value = True
        mock_minio_class.return_value = mock_client
        
        result = s3_loader.upload_pdf_to_s3("/path/to/test.pdf", "test-key.pdf")
        
        assert result is True
        mock_client.fput_object.assert_called_once_with(
            "test-bucket", "test-key.pdf", "/path/to/test.pdf", content_type='application/pdf'
        )

    @patch('data_processing.s3_loader.Minio')
    def test_upload_pdf_to_s3_error(self, mock_minio_class, s3_loader):
        """Test PDF upload error handling."""
        mock_client = Mock()
        mock_client.bucket_exists.return_value = True
        mock_client.fput_object.side_effect = Exception("Upload failed")
        mock_minio_class.return_value = mock_client
        
        result = s3_loader.upload_pdf_to_s3("/path/to/test.pdf", "test-key.pdf")
        
        assert result is False
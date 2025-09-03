"""Integration tests for S3 PDF loader with MinIO."""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock

import pytest
from minio import Minio
from minio.error import S3Error

from configs import DocumentsConfig, VectorStoreConfig
from configs.app_settings import Settings
from data_processing.s3_loader import S3PDFLoader


@pytest.fixture(scope="session")
def minio_settings():
    """Create settings for MinIO integration tests."""
    settings = Settings()
    # Use test-specific bucket to avoid conflicts
    settings.s3_bucket_name = "test-integration-bucket"
    return settings


@pytest.fixture(scope="session")
def minio_client(minio_settings):
    """Create MinIO client for integration tests."""
    try:
        client = Minio(
            endpoint=minio_settings.s3_endpoint,
            access_key=minio_settings.s3_access_key,
            secret_key=minio_settings.s3_secret_key,
            secure=minio_settings.s3_secure,
            region=minio_settings.s3_region,
        )
        
        # Test connection
        client.list_buckets()
        return client
        
    except Exception as e:
        pytest.skip(f"MinIO not available for integration tests: {e}")


@pytest.fixture(scope="session")
def test_bucket(minio_client, minio_settings):
    """Create and cleanup test bucket."""
    bucket_name = minio_settings.s3_bucket_name
    
    # Create bucket if it doesn't exist
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
    
    yield bucket_name
    
    # Cleanup: remove all objects and bucket
    try:
        objects = minio_client.list_objects(bucket_name, recursive=True)
        for obj in objects:
            minio_client.remove_object(bucket_name, obj.object_name)
        minio_client.remove_bucket(bucket_name)
    except Exception:
        pass  # Best effort cleanup


@pytest.fixture
def sample_pdf():
    """Create a sample PDF file for testing."""
    # Create a simple PDF using reportlab if available, otherwise skip
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            pdf_path = tmp_file.name
            
        # Create a simple PDF
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.drawString(100, 750, "Test PDF Document")
        c.drawString(100, 730, "This is a test document for S3 integration tests.")
        c.drawString(100, 710, "Page 1 content.")
        c.showPage()
        
        c.drawString(100, 750, "Page 2")
        c.drawString(100, 730, "This is the second page of the test document.")
        c.showPage()
        
        c.save()
        
        yield pdf_path
        
        # Cleanup
        try:
            Path(pdf_path).unlink()
        except Exception:
            pass
            
    except ImportError:
        pytest.skip("reportlab not available for PDF generation")


@pytest.fixture
def s3_loader(minio_settings):
    """Create S3 loader for integration tests."""
    documents_config = DocumentsConfig(file_path="")
    vector_store_config = VectorStoreConfig()
    chunking_config = Mock()
    chunking_config.type = "recursive_character"
    
    return S3PDFLoader(
        documents_config=documents_config,
        chunking_config=chunking_config,
        vector_store_config=vector_store_config,
        settings=minio_settings,
    )


@pytest.mark.integration
class TestS3IntegrationTests:
    """Integration tests for S3 PDF loader."""

    def test_bucket_operations(self, s3_loader, test_bucket):
        """Test bucket creation and existence checking."""
        # Bucket should exist after fixture setup
        assert s3_loader.minio_client.bucket_exists(test_bucket)
        
        # Test ensure_bucket_exists with existing bucket
        s3_loader._ensure_bucket_exists()  # Should not raise

    def test_upload_and_list_operations(self, s3_loader, sample_pdf, test_bucket):
        """Test uploading PDF and listing objects."""
        s3_key = "test-equipment/manual.pdf"
        
        # Upload PDF
        success = s3_loader.upload_pdf_to_s3(sample_pdf, s3_key)
        assert success is True
        
        # List objects
        objects = s3_loader.list_s3_objects()
        assert s3_key in objects
        
        # List with prefix
        objects_with_prefix = s3_loader.list_s3_objects("test-equipment/")
        assert s3_key in objects_with_prefix
        
        # List with non-matching prefix
        objects_no_match = s3_loader.list_s3_objects("non-existent/")
        assert len(objects_no_match) == 0

    def test_load_documents_from_s3(self, s3_loader, sample_pdf, test_bucket):
        """Test loading documents from S3."""
        s3_key = "equipment1/manuals/test-manual.pdf"
        
        # Upload test PDF
        success = s3_loader.upload_pdf_to_s3(sample_pdf, s3_key)
        assert success is True
        
        # Wait a moment for upload to complete
        time.sleep(1)
        
        # Load documents
        documents = s3_loader.load_documents()
        
        assert len(documents) > 0
        
        # Check document metadata
        doc = documents[0]
        assert doc.metadata["file_name"] == "test-manual.pdf"
        assert doc.metadata["s3_key"] == s3_key
        assert doc.metadata["s3_bucket"] == test_bucket
        assert doc.metadata["equipment"] == "equipment1"
        assert doc.metadata["document_source"] == "s3"
        assert "source" in doc.metadata
        
        # Check content
        assert len(doc.page_content) > 0
        assert "Test PDF Document" in doc.page_content

    def test_load_documents_with_equipment_filter(self, s3_loader, sample_pdf, test_bucket):
        """Test loading documents with equipment filter."""
        # Upload PDFs for different equipment
        s3_loader.upload_pdf_to_s3(sample_pdf, "equipment1/manual1.pdf")
        s3_loader.upload_pdf_to_s3(sample_pdf, "equipment2/manual2.pdf")
        
        time.sleep(1)
        
        # Load documents for specific equipment
        docs_eq1 = s3_loader.load_documents(equipment="equipment1")
        docs_eq2 = s3_loader.load_documents(equipment="equipment2")
        
        # Should find documents for each equipment
        assert len(docs_eq1) > 0
        assert len(docs_eq2) > 0
        
        # Check equipment metadata
        assert all(doc.metadata["equipment"] == "equipment1" for doc in docs_eq1)
        assert all(doc.metadata["equipment"] == "equipment2" for doc in docs_eq2)

    def test_process_documents_integration(self, s3_loader, sample_pdf, test_bucket):
        """Test full document processing pipeline."""
        s3_key = "test-equipment/integration-test.pdf"
        
        # Upload and load documents
        s3_loader.upload_pdf_to_s3(sample_pdf, s3_key)
        time.sleep(1)
        
        documents = s3_loader.load_documents()
        assert len(documents) > 0
        
        # Create a real chunking config for testing
        from configs.rag_configs import RecursiveCharacterSplitterConfig
        real_chunking_config = RecursiveCharacterSplitterConfig(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        # Replace the mock config with real one
        s3_loader.chunking_config = real_chunking_config
        s3_loader._text_splitter = None  # Reset to force re-initialization
        
        # Process documents - should work now
        processed_docs = s3_loader.process_documents(documents)
        assert len(processed_docs) >= len(documents)  # Should have same or more chunks

    def test_error_handling_nonexistent_file(self, s3_loader):
        """Test error handling when trying to upload non-existent file."""
        success = s3_loader.upload_pdf_to_s3("/non/existent/file.pdf", "test.pdf")
        assert success is False

    def test_load_documents_empty_bucket(self, s3_loader, test_bucket):
        """Test loading documents from empty bucket."""
        # Ensure bucket is empty
        objects = s3_loader.minio_client.list_objects(test_bucket, recursive=True)
        for obj in objects:
            s3_loader.minio_client.remove_object(test_bucket, obj.object_name)
        
        # Try to load documents
        with pytest.raises(ValueError, match="No PDF files found"):
            s3_loader.load_documents()

    def test_connection_error_handling(self, minio_settings):
        """Test handling of connection errors."""
        # Create loader with invalid endpoint
        bad_settings = Settings()
        bad_settings.s3_endpoint = "invalid-endpoint:9999"
        bad_settings.s3_bucket_name = "test-bucket"
        
        documents_config = DocumentsConfig(file_path="")
        vector_store_config = VectorStoreConfig()
        chunking_config = Mock()
        
        bad_loader = S3PDFLoader(
            documents_config=documents_config,
            chunking_config=chunking_config,
            vector_store_config=vector_store_config,
            settings=bad_settings,
        )
        
        # This should fail when trying to access MinIO
        with pytest.raises(Exception):
            bad_loader.load_documents()


@pytest.mark.integration
def test_s3_loader_factory_integration(minio_settings):
    """Test S3 loader creation through factory."""
    from data_processing.loader_factory import DataLoaderFactory
    from configs.app_settings import DocumentSource
    
    # Set document source to S3
    minio_settings.document_source = DocumentSource.S3
    
    documents_config = DocumentsConfig(file_path="")
    vector_store_config = VectorStoreConfig()
    chunking_config = Mock()
    
    loader = DataLoaderFactory.create_documents_loader(
        documents_config=documents_config,
        chunking_config=chunking_config,
        vector_store_config=vector_store_config,
        settings=minio_settings,
    )
    
    assert isinstance(loader, S3PDFLoader)
    assert loader.settings == minio_settings
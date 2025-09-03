#!/usr/bin/env python3
"""
Test script to verify S3 functionality end-to-end.
This script tests the complete S3 integration including MinIO setup.
"""

import logging
import os
import tempfile
from pathlib import Path

from configs import DocumentsConfig, VectorStoreConfig
from configs.app_settings import DocumentSource, Settings
from data_processing.loader_factory import DataLoaderFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_pdf() -> str:
    """
    Create a simple test PDF file.
    
    Returns:
        Path to the created PDF file
    """
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            pdf_path = tmp_file.name
            
        # Create a simple PDF
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.drawString(100, 750, "Test Medical Equipment Manual")
        c.drawString(100, 730, "This is a test document for S3 integration.")
        c.drawString(100, 710, "Equipment: Test Device Model X1")
        c.drawString(100, 690, "Page 1 - Installation Instructions")
        c.showPage()
        
        c.drawString(100, 750, "Page 2 - Maintenance Guide")
        c.drawString(100, 730, "Regular maintenance procedures:")
        c.drawString(100, 710, "1. Clean the device weekly")
        c.drawString(100, 690, "2. Check calibration monthly")
        c.showPage()
        
        c.save()
        logger.info(f"Created test PDF: {pdf_path}")
        return pdf_path
        
    except ImportError:
        logger.error("reportlab not available. Cannot create test PDF.")
        raise


def test_s3_connection(settings: Settings) -> bool:
    """
    Test basic S3 connection.
    
    Args:
        settings: Application settings
        
    Returns:
        True if connection successful
    """
    try:
        from minio import Minio
        
        client = Minio(
            endpoint=settings.s3_endpoint,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            secure=settings.s3_secure,
            region=settings.s3_region,
        )
        
        # Test connection by listing buckets
        buckets = client.list_buckets()
        logger.info(f"✅ S3 connection successful. Found {len(buckets)} buckets.")
        return True
        
    except Exception as e:
        logger.error(f"❌ S3 connection failed: {e}")
        return False


def test_s3_loader_creation(settings: Settings) -> bool:
    """
    Test S3 loader creation through factory.
    
    Args:
        settings: Application settings
        
    Returns:
        True if loader created successfully
    """
    try:
        documents_config = DocumentsConfig(file_path="")
        vector_store_config = VectorStoreConfig()
        
        # Mock chunking config
        class MockChunkingConfig:
            type = "recursive_character"
        
        chunking_config = MockChunkingConfig()
        
        # Set document source to S3
        settings.document_source = DocumentSource.S3
        
        loader = DataLoaderFactory.create_documents_loader(
            documents_config=documents_config,
            chunking_config=chunking_config,
            vector_store_config=vector_store_config,
            settings=settings,
        )
        
        from data_processing.s3_loader import S3PDFLoader
        assert isinstance(loader, S3PDFLoader)
        
        logger.info("✅ S3 loader created successfully through factory")
        return True
        
    except Exception as e:
        logger.error(f"❌ S3 loader creation failed: {e}")
        return False


def test_pdf_upload_and_load(settings: Settings) -> bool:
    """
    Test PDF upload to S3 and subsequent loading.
    
    Args:
        settings: Application settings
        
    Returns:
        True if upload and load successful
    """
    pdf_path = None
    try:
        # Create test PDF
        pdf_path = create_test_pdf()
        
        # Create S3 loader
        documents_config = DocumentsConfig(file_path="")
        vector_store_config = VectorStoreConfig()
        
        class MockChunkingConfig:
            type = "recursive_character"
        
        chunking_config = MockChunkingConfig()
        settings.document_source = DocumentSource.S3
        
        loader = DataLoaderFactory.create_documents_loader(
            documents_config=documents_config,
            chunking_config=chunking_config,
            vector_store_config=vector_store_config,
            settings=settings,
        )
        
        # Upload PDF
        s3_key = "test-equipment/test-manual.pdf"
        upload_success = loader.upload_pdf_to_s3(pdf_path, s3_key)
        
        if not upload_success:
            logger.error("❌ PDF upload failed")
            return False
        
        logger.info("✅ PDF uploaded successfully")
        
        # List S3 objects
        objects = loader.list_s3_objects()
        if s3_key not in objects:
            logger.error(f"❌ Uploaded file not found in S3. Objects: {objects}")
            return False
        
        logger.info(f"✅ PDF found in S3. Total objects: {len(objects)}")
        
        # Load documents
        documents = loader.load_documents()
        
        if not documents:
            logger.error("❌ No documents loaded from S3")
            return False
        
        logger.info(f"✅ Loaded {len(documents)} documents from S3")
        
        # Verify document metadata
        doc = documents[0]
        expected_metadata_keys = [
            "file_name", "s3_key", "s3_bucket", "equipment", 
            "document_source", "source", "page_number"
        ]
        
        for key in expected_metadata_keys:
            if key not in doc.metadata:
                logger.error(f"❌ Missing metadata key: {key}")
                return False
        
        logger.info("✅ Document metadata validation passed")
        
        # Verify content
        if "Test Medical Equipment Manual" not in doc.page_content:
            logger.error("❌ Expected content not found in document")
            return False
        
        logger.info("✅ Document content validation passed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ PDF upload and load test failed: {e}")
        return False
        
    finally:
        # Cleanup
        if pdf_path and Path(pdf_path).exists():
            try:
                Path(pdf_path).unlink()
                logger.info(f"Cleaned up test PDF: {pdf_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup test PDF: {e}")


def test_equipment_filtering(settings: Settings) -> bool:
    """
    Test equipment-based document filtering.
    
    Args:
        settings: Application settings
        
    Returns:
        True if filtering works correctly
    """
    pdf_path1 = None
    pdf_path2 = None
    
    try:
        # Create test PDFs
        pdf_path1 = create_test_pdf()
        pdf_path2 = create_test_pdf()
        
        # Create S3 loader
        documents_config = DocumentsConfig(file_path="")
        vector_store_config = VectorStoreConfig()
        
        class MockChunkingConfig:
            type = "recursive_character"
        
        chunking_config = MockChunkingConfig()
        settings.document_source = DocumentSource.S3
        
        loader = DataLoaderFactory.create_documents_loader(
            documents_config=documents_config,
            chunking_config=chunking_config,
            vector_store_config=vector_store_config,
            settings=settings,
        )
        
        # Upload PDFs for different equipment
        loader.upload_pdf_to_s3(pdf_path1, "equipment1/manual1.pdf")
        loader.upload_pdf_to_s3(pdf_path2, "equipment2/manual2.pdf")
        
        # Test filtering for equipment1
        docs_eq1 = loader.load_documents(equipment="equipment1")
        if not docs_eq1:
            logger.error("❌ No documents found for equipment1")
            return False
        
        # Verify all documents are for equipment1
        for doc in docs_eq1:
            if doc.metadata.get("equipment") != "equipment1":
                logger.error(f"❌ Wrong equipment in metadata: {doc.metadata.get('equipment')}")
                return False
        
        logger.info(f"✅ Equipment filtering works. Found {len(docs_eq1)} docs for equipment1")
        
        # Test filtering for equipment2
        docs_eq2 = loader.load_documents(equipment="equipment2")
        if not docs_eq2:
            logger.error("❌ No documents found for equipment2")
            return False
        
        # Verify all documents are for equipment2
        for doc in docs_eq2:
            if doc.metadata.get("equipment") != "equipment2":
                logger.error(f"❌ Wrong equipment in metadata: {doc.metadata.get('equipment')}")
                return False
        
        logger.info(f"✅ Equipment filtering works. Found {len(docs_eq2)} docs for equipment2")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Equipment filtering test failed: {e}")
        return False
        
    finally:
        # Cleanup
        for pdf_path in [pdf_path1, pdf_path2]:
            if pdf_path and Path(pdf_path).exists():
                try:
                    Path(pdf_path).unlink()
                except Exception:
                    pass


def main():
    """Run all S3 functionality tests."""
    logger.info("🚀 Starting S3 functionality tests...")
    
    # Load settings
    settings = Settings()
    logger.info(f"Using S3 endpoint: {settings.s3_endpoint}")
    logger.info(f"Using S3 bucket: {settings.s3_bucket_name}")
    
    tests = [
        ("S3 Connection", lambda: test_s3_connection(settings)),
        ("S3 Loader Creation", lambda: test_s3_loader_creation(settings)),
        ("PDF Upload and Load", lambda: test_pdf_upload_and_load(settings)),
        ("Equipment Filtering", lambda: test_equipment_filtering(settings)),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running test: {test_name}")
        try:
            if test_func():
                logger.info(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
    
    logger.info(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All S3 functionality tests passed!")
        return 0
    else:
        logger.error("💥 Some tests failed. Check the logs above.")
        return 1


if __name__ == "__main__":
    exit(main())
#!/usr/bin/env python3
"""
Tests for document chunking functionality.
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document

from data_processing.splitters import (
    PageBasedTextSplitter,
    RecursiveCharacterTextSplitterWrapper,
    SemanticTextSplitter,
    TextSplitterFactory,
)


def debug_chunking(
    documents: List[Document],
    splitter,
    target_page: Optional[int] = None,
    target_file: Optional[str] = None,
) -> List[Document]:
    """
    Debug function to test chunking on specific documents.

    This function is for testing and debugging purposes only.
    It should not be used in production code.

    Args:
        documents: List of documents to test
        splitter: Text splitter instance
        target_page: Optional page number to focus on
        target_file: Optional file name to focus on

    Returns:
        List of chunks created by the splitter
    """
    logger = logging.getLogger(__name__)
    logger.info("=== CHUNKING DEBUG ===")

    # Filter documents if target specified
    if target_page or target_file:
        filtered_docs = []
        for doc in documents:
            if target_page and doc.metadata.get("page_number") == target_page:
                filtered_docs.append(doc)
            elif target_file and target_file in doc.metadata.get("file_name", ""):
                filtered_docs.append(doc)
        if filtered_docs:
            documents = filtered_docs
            logger.info(f"Filtered to {len(documents)} documents")

    # Show document info
    for i, doc in enumerate(documents):
        logger.info(
            f"Document {i+1}: {doc.metadata.get('file_name', 'unknown')} - "
            f"Page {doc.metadata.get('page_number', 'unknown')}"
        )
        logger.info(f"Content preview: {doc.page_content[:200]}...")

    # Test chunking
    chunks = splitter.split_documents(documents)

    logger.info("=== CHUNKING RESULTS ===")
    logger.info(f"Input: {len(documents)} documents")
    logger.info(f"Output: {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        logger.info(f"Chunk {i+1}:")
        logger.info(f"  File: {chunk.metadata.get('file_name', 'unknown')}")
        logger.info(f"  Page: {chunk.metadata.get('page_number', 'unknown')}")
        logger.info(f"  Section: {chunk.metadata.get('section_title', 'N/A')}")
        logger.info(f"  Words: {len(chunk.page_content.split())}")
        logger.info(f"  Content: {chunk.page_content[:300]}...")
        logger.info("---")

    return chunks


def test_page_based_splitter():
    """Test the PageBasedTextSplitter with sample documents."""
    # Create sample documents
    docs = [
        Document(
            page_content="This is page 1 content. It has multiple paragraphs.\n\n"
            "Second paragraph with more content.",
            metadata={"file_name": "test.pdf", "page_number": 1},
        ),
        Document(
            page_content="This is page 2 content. Different content here.\n\n"
            "Another paragraph on page 2.",
            metadata={"file_name": "test.pdf", "page_number": 2},
        ),
    ]

    splitter = PageBasedTextSplitter(min_chunk_words=5)
    chunks = splitter.split_documents(docs)

    assert len(chunks) > 0, "Should create at least one chunk"
    assert all(
        chunk.metadata.get("page_number") for chunk in chunks
    ), "All chunks should have page numbers"

    return chunks


def test_semantic_splitter():
    """Test the SemanticTextSplitter with sample documents."""
    docs = [
        Document(
            page_content="3.6.4 Test Section\n\n"
            "This is the content of section 3.6.4.\n\n"
            "3.6.5 Another Section\n\nContent for section 3.6.5.",
            metadata={"file_name": "test.pdf", "page_number": 1},
        )
    ]

    splitter = SemanticTextSplitter(
        model_name="sentence-transformers/all-MiniLM-L6-v2", min_chunk_words=10
    )
    chunks = splitter.split_documents(docs)

    assert len(chunks) > 0, "Should create at least one chunk"

    return chunks


def test_text_splitter_factory():
    """Test the TextSplitterFactory creates correct splitters."""
    from configs.rag_configs import (
        PageBasedSplitterConfig,
        RecursiveCharacterSplitterConfig,
        SemanticSplitterConfig,
    )

    # Test page-based config
    page_config = PageBasedSplitterConfig(min_chunk_words=10)
    page_splitter = TextSplitterFactory.create(page_config)
    assert isinstance(page_splitter, PageBasedTextSplitter)

    # Test recursive character config
    rec_config = RecursiveCharacterSplitterConfig(chunk_size=1000, chunk_overlap=200)
    rec_splitter = TextSplitterFactory.create(rec_config)
    assert isinstance(rec_splitter, RecursiveCharacterTextSplitterWrapper)

    # Test semantic config
    sem_config = SemanticSplitterConfig(min_chunk_words=10)
    sem_splitter = TextSplitterFactory.create(sem_config)
    assert isinstance(sem_splitter, SemanticTextSplitter)


if __name__ == "__main__":
    # Set up logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("Running chunking tests...")

    # Run tests
    test_page_based_splitter()
    test_semantic_splitter()
    test_text_splitter_factory()

    print("All tests passed!")

from typing import List

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class Answer(BaseModel):
    """Represents an answer with source documents."""

    answer_text: str = Field(..., description="Generated LLM text response")
    source_chunks: List[Document] = Field(
        ..., description="List of chunks used as context"
    )
    response_type: str = Field(
        ..., description="Type of response: 'ANSWER' or 'CLARIFICATION'"
    )

    def __repr__(self) -> str:
        return (
            f"Answer(answer_text='{self.answer_text[:50]}...',"
            f"sources={len(self.source_chunks)}, "
            f"needs_clarification={self.needs_clarification})"
        )


class DocumentChunk(BaseModel):
    """Represents a chunk of a document with metadata."""

    document_id: str = Field(
        ..., description="Unique identifier of the source document"
    )
    filename: str = Field(..., description="Name of the source file")
    page_number: int = Field(..., description="Page number in the document")
    content: str = Field(..., description="Chunk text content")
    score: float = Field(
        1.0, description="Relevance assessment from a retriever or reranker"
    )
    chunk_index: int = Field(..., description="Chunk number")

    def __repr__(self) -> str:
        return (
            f"DocumentChunk(doc_id='{self.document_id}', "
            f"page={self.page_number}, score={self.score:.5f})"
        )

import datetime
import re
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from pydantic import BaseModel, Field, field_validator


class ResponseType(str, Enum):
    """Enum for response types"""

    ANSWER = "ANSWER"
    CLARIFICATION = "CLARIFICATION"


class ChatMessage(BaseModel):
    """Model for a chat message"""

    role: str = Field(
        ...,
        description="Role of the message sender (user or assistant)",
        pattern="^(user|assistant)$",
    )
    content: str = Field(
        ..., description="Content of the message", min_length=1, max_length=10_000
    )


class QueryRequest(BaseModel):
    """Request model for document query"""

    request_id: str = Field(
        ...,
        description="Unique request identifier for tracing",
        min_length=1,
        max_length=100,
    )
    question: str = Field(
        ..., description="User question to be answered", min_length=1, max_length=10_000
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier for chat history management",
        min_length=1,
        max_length=100,
    )
    equipment: Optional[str] = Field(
        default=None,
        description="Equipment identifier or name for filtering relevant documents.",
    )

    @field_validator("question")
    def validate_question(cls, v):
        """Validate and sanitize question input."""
        if not v.strip():
            raise ValueError("Question cannot be empty or whitespace only")
        return v.strip()

    @field_validator("request_id")
    def validate_request_id(cls, v):
        """Validate request ID format"""
        if not re.match(r"^[a-zA-Z0-9\-_]+$", v):
            raise ValueError("Invalid request_id format")
        return v


class SourceDocument(BaseModel):
    """Source document information"""

    document_id: str = Field(
        ..., description="Unique identifier for the document chunk"
    )
    filename: str = Field(..., description="Original source filename or URI")
    page_number: Optional[int] = Field(
        None, ge=1, description="Page number within document, if paginated"
    )
    chunk_index: str = Field(
        ..., description="Logical chunk index ot path within the document"
    )
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Relevance score (0-1) assigned by retriever"
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=1_000_000,
        description="Text payload of this chunk",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional metadata",
    )


class QueryResponse(BaseModel):
    """Response model for document query"""

    request_id: str = Field(
        ...,
        description="Unique request identifier for tracing",
        min_length=1,
        max_length=100,
    )
    answer: str = Field(
        ..., description="Answer from the system", min_length=1, max_length=10_000
    )
    type: str = Field(
        default="ANSWER",
        description="Type of response: ANSWER or CLARIFICATION",
        pattern="^(ANSWER|CLARIFICATION)$",
    )
    sources: Optional[List[Document]] = Field(
        None, description="List of source documents used to answer the question"
    )

    @field_validator("answer")
    def validate_question(cls, v):
        """Validate answer content."""
        if not v.strip():
            raise ValueError("Answer cannot be empty")
        return v.strip()

    @field_validator("sources")
    def validate_sources(cls, v):
        """Validate answer content."""
        if len(v) > 100:
            raise ValueError("Too many source documents")
        return v


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(
        ...,
        description="Overall service health status: healthy, degraded, or unhealthy",
    )
    version: str = Field(..., description="Service semantic version")
    components: Dict[str, str] = Field(
        ..., description="Mapping of individual component names to their health status"
    )
    timestamp: datetime.datetime = Field(
        default_factory=datetime.datetime.now(datetime.timezone.utc),
        description="UTC time when health was checked",
    )
    uptime_seconds: float = Field(
        default=0.0, ge=0.0, description="Total seconds the service has been up"
    )

    @field_validator("status")
    def validate_status(cls, v):
        """Validate health status."""
        if v not in {"healthy", "unhealthy", "degraded"}:
            raise ValueError("Invalid health status")
        return v

    @field_validator("timestamp")
    def validate_timestamp(cls, v) -> datetime.datetime:
        """Reject timestamps set in the future.."""
        if v > datetime.datetime.now(datetime.timezone.utc):
            raise ValueError("timestamp cannot be in the future")
        return v

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "components": {
                    "database": {"status": "healthy", "latency_ms": 15},
                    "llm": {"status": "healthy", "latency_ms": 850},
                    "vector_store": {"status": "healthy", "size": 12500},
                },
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }


class ClarificationRequest(BaseModel):
    """Request model for clarification question generation"""

    request_id: str = Field(
        ...,
        description="Unique request identifier for tracing",
        min_length=1,
        max_length=100,
    )
    question: str = Field(
        default="",
        description="User question that needs clarification "
        "(optional, can be empty if using only session context)",
        max_length=10_000,
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier for chat history context",
        min_length=1,
        max_length=100,
    )

    @field_validator("question")
    def validate_question(cls, v):
        """Validate and sanitize question input."""
        if v is None:
            return ""
        return v.strip()

    @field_validator("request_id")
    def validate_request_id(cls, v):
        """Validate request ID format"""
        if not re.match(r"^[a-zA-Z0-9\-_]+$", v):
            raise ValueError("Invalid request_id format")
        return v


class ClarificationResponse(BaseModel):
    """Response model for clarification question"""

    request_id: str = Field(
        ...,
        description="Unique request identifier for tracing",
        min_length=1,
        max_length=100,
    )
    clarification_question: str = Field(
        ...,
        description="Generated clarification question",
        min_length=1,
        max_length=2000,
    )
    session_id: Optional[str] = Field(
        None,
        description="Session identifier used for context",
        min_length=1,
        max_length=100,
    )


class ErrorResponse(BaseModel):
    """Standardized error response model."""

    error_code: str = Field(..., description="Error code for client handling")
    message: str = Field(..., description="User-friendly error message")
    request_id: Optional[str] = Field(None, description="Request identifier")
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
        description="UTC time when the health check was performed",
    )

    class Config:
        schema_extra = {
            "example": {
                "error_code": "VALIDATION_ERROR",
                "message": "Invalid input parameters",
                "request_id": "req-123456",
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }


class SessionStatsResponse(BaseModel):
    """Response model for session statistics"""

    session_id: str = Field(..., description="Session identifier")
    exists: bool = Field(..., description="Whether session exists")
    memory_type: Optional[str] = Field(None, description="Type of memory used")
    message_count: int = Field(..., description="Number of messages in session")
    estimated_tokens: int = Field(..., description="Estimated token count")
    max_token_limit: Optional[int] = Field(None, description="Maximum token limit")
    window_size: Optional[int] = Field(None, description="Memory window size")


class ClearSessionResponse(BaseModel):
    """Response model for session clearing"""

    session_id: str = Field(..., description="Session identifier")
    cleared: bool = Field(..., description="Whether session was cleared")


class ActiveSessionsResponse(BaseModel):
    """Response model for active sessions list"""

    active_sessions: List[str] = Field(..., description="List of active session IDs")
    count: int = Field(..., description="Number of active sessions")


class CleanupResponse(BaseModel):
    """Response model for cleanup operation"""

    cleaned_sessions: int = Field(..., description="Number of sessions cleaned up")

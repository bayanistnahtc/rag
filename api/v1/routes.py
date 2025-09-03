import datetime
import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder

from api.dependencies import get_rag_service
from api.schemas import (
    ActiveSessionsResponse,
    ClarificationRequest,
    ClarificationResponse,
    CleanupResponse,
    ClearSessionResponse,
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SessionStatsResponse,
)
from rag.models import Answer
from rag.orchestrator import RAGOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["RAG"])


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Query equipment documentation",
    description="Submit a question about medical equipment and "
    "receive AI-generated answers with source references.",
)
async def query_documents(
    request: QueryRequest, service: RAGOrchestrator = Depends(get_rag_service)
):
    """
    Process a document query and return AI-generated answer with sources.

    Args:
        request: Query request with question and metadata
        rag_service: RAG orchestrator service instance

    Returns:
        QueryResponse: Generated answer with source documents

    Raises:
        HTTPException: For various error conditions
    """
    start_time = time.time()

    logger.info(
        "Processing query request:",
        extra={
            "request_id": request.request_id,
            "question_length": len(request.question),
            "session_id": request.session_id,
        },
    )
    try:
        # Process the query using session-based history management
        result: Answer = service.invoke(
            query=request.question,
            session_id=request.session_id,
            equipment=request.equipment,
        )

        processing_time = int((time.time() - start_time) * 1000)
        response = QueryResponse(
            request_id=request.request_id,
            answer=result.answer_text,
            sources=result.source_chunks,
            type=result.response_type,
        )
        logger.info(
            "Query processed successfully",
            extra={
                "request_id": request.request_id,
                "processing_time_ms": processing_time,
                "sources_count": len(result.source_chunks),
                "answer_length": len(result.answer_text),
                "response_type": result.response_type,
            },
        )
        return response
    except ValueError as e:
        logger.warning(
            f"Validation error in query processing: {e}",
            extra={"request_id": request.request_id, "error": str(e)},
        )
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error_code="VALIDATION_ERROR",
                message="Invalid input parameters",
                request_id=request.request_id,
            ).model_dump(),
        )

    except TimeoutError as e:
        logger.error(
            f"Query processing timeout: {e}",
            extra={
                "request_id": request.request_id,
                "processing_time_ms": int((time.time() - start_time) * 1000),
            },
        )
        raise HTTPException(
            status_code=504,
            detail=ErrorResponse(
                error_code="PROCESSING_TIMEOUT",
                message="Query processing timed out",
                request_id=request.request_id,
            ).model_dump(),
        )

    except Exception as e:
        logger.error(
            "Unexpected error in query processing",
            extra={
                "request_id": request.request_id,
                "error_type": type(e).__name__,
                "processing_time_ms": int((time.time() - start_time) * 1000),
            },
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=jsonable_encoder(
                ErrorResponse(
                    error_code="INTERNAL_ERROR",
                    message="An unexpected error occurred",
                    request_id=request.request_id,
                ).model_dump()
            ),
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint",
    description="Check the health status of the RAG service and its components.",
)
async def health_check(
    rag_service: RAGOrchestrator = Depends(get_rag_service),
) -> HealthResponse:
    """
    Perform health check of the RAG service.

    Args:
        rag_service: RAG orchestrator service instance

    Returns:
        HealthResponse: Service health status
    """
    # Track start time for uptime calculation
    # start_time = time.time()

    try:
        # Check component health
        components = await rag_service.health_check()

        # Convert components to proper format (Dict[str, str])
        formatted_components = {}
        for key, value in components.items():
            if isinstance(value, dict):
                formatted_components[key] = str(value)
            else:
                formatted_components[key] = str(value)

        overall_status = (
            "healthy"
            if all(comp.get("status") == "healthy" for comp in components.values())
            else "degraded"
        )

        # Create response with explicit timestamp and uptime
        from fastapi.responses import JSONResponse

        return JSONResponse(
            content=jsonable_encoder(
                HealthResponse(
                    status=overall_status,
                    version="1.0.0",
                    components=formatted_components,
                    timestamp=datetime.datetime.now(datetime.timezone.utc),
                    uptime_seconds=0.0,  # We don't track actual uptime yet
                )
            )
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)

        # Create error response with explicit timestamp and uptime
        from fastapi.responses import JSONResponse

        return JSONResponse(
            content=jsonable_encoder(
                HealthResponse(
                    status="unhealthy",
                    version="1.0.0",
                    components={"error": "Health check failed: " + str(e)},
                    timestamp=datetime.datetime.now(datetime.timezone.utc),
                    uptime_seconds=0.0,
                )
            )
        )


@router.post(
    "/reindex",
    summary="Reindex documents",
    description="Force reindexing of all documents from "
    "the configured source (local or S3).",
    responses={
        200: {"description": "Reindexing completed successfully"},
        500: {"model": ErrorResponse, "description": "Reindexing failed"},
    },
)
async def reindex_documents(service: RAGOrchestrator = Depends(get_rag_service)):
    """
    Force reindexing of all documents.

    This endpoint will reload all documents from the configured source
    (local files or S3) and rebuild the vector indexes.

    Args:
        service: RAG orchestrator service instance

    Returns:
        Success message with reindexing statistics

    Raises:
        HTTPException: If reindexing fails
    """
    start_time = time.time()

    try:
        logger.info("Starting document reindexing...")

        # Force rebuild of indexes
        service.initialize(force_rebuild_storage=True)

        processing_time = int((time.time() - start_time) * 1000)

        # Get index statistics
        index_size = 0
        if service.vector_store and service.vector_store.index_exists():
            index_size = service.vector_store.get_index_size()

        logger.info(
            "Document reindexing completed successfully",
            extra={
                "processing_time_ms": processing_time,
                "index_size": index_size,
            },
        )

        return {
            "status": "success",
            "message": "Documents reindexed successfully",
            "processing_time_ms": processing_time,
            "index_size": index_size,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        logger.error(
            "Document reindexing failed",
            extra={
                "error_type": type(e).__name__,
                "processing_time_ms": processing_time,
            },
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error_code="REINDEX_ERROR",
                message=f"Document reindexing failed: {str(e)}",
                request_id=None,
            ).model_dump(),
        )


# Session management endpoints
@router.get(
    "/sessions",
    response_model=ActiveSessionsResponse,
    summary="Get active sessions",
    description="Retrieve list of all active chat sessions.",
)
async def get_active_sessions(service: RAGOrchestrator = Depends(get_rag_service)):
    """
    Get list of all active session IDs.

    Returns:
        ActiveSessionsResponse: List of active sessions
    """
    try:
        active_sessions = service.get_active_sessions()
        return ActiveSessionsResponse(
            active_sessions=active_sessions, count=len(active_sessions)
        )
    except Exception as e:
        logger.error(f"Failed to get active sessions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error_code="SESSION_ERROR",
                message="Failed to retrieve active sessions",
            ).model_dump(),
        )


@router.get(
    "/sessions/{session_id}",
    response_model=SessionStatsResponse,
    summary="Get session statistics",
    description="Retrieve statistics for a specific chat session.",
)
async def get_session_stats(
    session_id: str, service: RAGOrchestrator = Depends(get_rag_service)
):
    """
    Get statistics for a specific session.

    Args:
        session_id: Session identifier

    Returns:
        SessionStatsResponse: Session statistics
    """
    try:
        stats = service.get_session_stats(session_id)
        return SessionStatsResponse(
            session_id=session_id,
            exists=stats.get("exists", False),
            memory_type=stats.get("memory_type"),
            message_count=stats.get("message_count", 0),
            estimated_tokens=stats.get("estimated_tokens", 0),
            max_token_limit=stats.get("max_token_limit"),
            window_size=stats.get("window_size"),
        )
    except Exception as e:
        logger.error(
            f"Failed to get session stats for {session_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error_code="SESSION_ERROR",
                message=f"Failed to retrieve session statistics for {session_id}",
            ).model_dump(),
        )


@router.delete(
    "/sessions/{session_id}",
    response_model=ClearSessionResponse,
    summary="Clear session history",
    description="Clear chat history for a specific session.",
)
async def clear_session_history(
    session_id: str, service: RAGOrchestrator = Depends(get_rag_service)
):
    """
    Clear chat history for a specific session.

    Args:
        session_id: Session identifier

    Returns:
        ClearSessionResponse: Clear operation result
    """
    try:
        cleared = service.clear_session_history(session_id)
        return ClearSessionResponse(session_id=session_id, cleared=cleared)
    except Exception as e:
        logger.error(f"Failed to clear session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error_code="SESSION_ERROR",
                message=f"Failed to clear session {session_id}",
            ).model_dump(),
        )


@router.post(
    "/sessions/cleanup",
    response_model=CleanupResponse,
    summary="Cleanup expired sessions",
    description="Clean up expired or inactive chat sessions.",
)
async def cleanup_sessions(service: RAGOrchestrator = Depends(get_rag_service)):
    """
    Clean up expired sessions based on TTL configuration.

    Returns:
        CleanupResponse: Cleanup operation result
    """
    try:
        cleaned_count = service.cleanup_expired_sessions()
        return CleanupResponse(cleaned_sessions=cleaned_count)
    except Exception as e:
        logger.error(f"Failed to cleanup sessions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error_code="SESSION_ERROR",
                message="Failed to cleanup sessions",
            ).model_dump(),
        )


# Clarification endpoint
@router.post(
    "/clarification_question",
    response_model=ClarificationResponse,
    summary="Generate clarification question",
    description="Generate a clarification question based on user's query and "
    "session context.",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def generate_clarification_question(
    request: ClarificationRequest, service: RAGOrchestrator = Depends(get_rag_service)
):
    """
    Generate a clarification question based on the user's query and session history.

    Args:
        request: Clarification request with question and session context
        service: RAG orchestrator service instance

    Returns:
        ClarificationResponse: Generated clarification question

    Raises:
        HTTPException: For various error conditions
    """
    start_time = time.time()

    logger.info(
        "Generating clarification question:",
        extra={
            "request_id": request.request_id,
            "question_length": len(request.question),
            "session_id": request.session_id,
        },
    )

    try:
        # Generate clarification question
        clarification_question = service.generate_clarification_question(
            question=request.question,
            session_id=request.session_id,
        )

        processing_time = int((time.time() - start_time) * 1000)

        response = ClarificationResponse(
            request_id=request.request_id,
            clarification_question=clarification_question,
            session_id=request.session_id,
        )

        logger.info(
            "Clarification question generated successfully",
            extra={
                "request_id": request.request_id,
                "processing_time_ms": processing_time,
                "clarification_length": len(clarification_question),
            },
        )
        return response

    except ValueError as e:
        logger.warning(
            f"Validation error in clarification generation: {e}",
            extra={"request_id": request.request_id, "error": str(e)},
        )
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error_code="VALIDATION_ERROR",
                message="Invalid input parameters",
                request_id=request.request_id,
            ).model_dump(),
        )

    except Exception as e:
        logger.error(
            "Unexpected error in clarification generation",
            extra={
                "request_id": request.request_id,
                "error_type": type(e).__name__,
                "processing_time_ms": int((time.time() - start_time) * 1000),
            },
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=jsonable_encoder(
                ErrorResponse(
                    error_code="INTERNAL_ERROR",
                    message="An unexpected error occurred",
                    request_id=request.request_id,
                ).model_dump()
            ),
        )

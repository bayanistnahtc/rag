import logging
from typing import Optional

from fastapi import HTTPException, Request, status

from rag.orchestrator import RAGOrchestrator

logger = logging.getLogger(__name__)


def get_rag_service(request: Request) -> RAGOrchestrator:
    """
    Dependancy injector for the application's RAG orchestrator service
    instance.

    Args:
        request (Request): The incoming FastAPI request object,
        used to access 'app.state'

    Returns:
        RAGOrchestrator: The shared RAGOrchestrator instance.

    Raises:
        HTTPException(503): If the RAGOrchestrator instance is not found.
    """

    rag_orchestrator: Optional[RAGOrchestrator] = getattr(
        request.app.state, "rag_pipeline", None
    )
    if not rag_orchestrator:
        logger.error("RAGOrchestrator not found in app.state.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service dependency 'RAGOrchestrator'is not available",
        )

    return rag_orchestrator

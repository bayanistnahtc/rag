"""
Medical RAG Service - Main FastAPI Application
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.v1.routes import router as query_router
from configs import RAGConfig, Settings
from rag.orchestrator import RAGOrchestrator

# from core.logging import setup_logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager for application lifespan events

    Args:
        app (FastAPI): The FastAPI application instance.
    """

    logger.info("Starting Medical RAG Service...")
    config = RAGConfig.from_env()
    settings = Settings()
    app.state.rag_pipeline = None

    try:
        logger.info("Initializing RAGPipeline...")
        rag_pipeline = RAGOrchestrator(config=config, settings=settings)
        rag_pipeline.initialize(force_rebuild_storage=True)

        app.state.rag_pipeline = rag_pipeline
    except Exception as e:
        logger.error(f"Error initializing RAGOrchestrator: {e}")
        raise RuntimeError(f"Failed to initialize RAGOrchestrator: {e}")

    logger.info(
        "Application startup sequence complete. Service is ready to accept requests."
    )
    yield

    logger.info("Medical RAG Service stopped")


# Create FastAPI app
app = FastAPI(
    title="Medical RAG Service",
    description="RAG service for medical equipment manuals",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn

    settings = Settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )

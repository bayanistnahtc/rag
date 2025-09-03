import logging
import uuid
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class MedRAGAPIClient:
    """Client for MedRAG backend API."""

    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize API client.

        Args:
            base_url: Base URL for the API (e.g., "http://localhost:8000/api/v1")
            timeout: Request timeout in seconds
        """
        # Normalize base URL to avoid accidental double prefixes
        self.base_url = base_url.rstrip("/")
        if "/api/v1" in self.base_url:
            logger.warning(
                "base_url should be host only (e.g., http://host:port). "
                "Stripping '/api/v1' from provided base_url."
            )
            self.base_url = self.base_url.split("/api/v1")[0].rstrip("/")

        self.timeout = timeout
        self.session = requests.Session()

        # Add basic retry strategy for transient errors
        retries = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[502, 503, 504],
            allowed_methods=[
                "GET",
                "POST",
                "PUT",
                "DELETE",
                "HEAD",
                "OPTIONS",
                "PATCH",
            ],
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Make HTTP request with proper error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., "/query/health")
            **kwargs: Additional arguments for requests

        Returns:
            Response object

        Raises:
            requests.RequestException: If request fails
        """
        # Build URL explicitly to preserve the provided base path
        url = f"{self.base_url.rstrip('/')}/api/v1/{endpoint.lstrip('/')}"

        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=kwargs.pop("timeout", self.timeout),
                **kwargs,
            )
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            # Try to extract response details for better diagnostics
            resp = getattr(e, "response", None)
            status = resp.status_code if resp is not None else None
            resp_text = None
            try:
                if resp is not None:
                    resp_text = resp.text
            except Exception:
                resp_text = "<unavailable>"
            logger.error(
                f"API request failed: {method} {url} - status={status}"
                f"error={e} response_text={resp_text}"
            )
            raise

    def health_check(self) -> Dict[str, Any]:
        """
        Check system health.

        Returns:
            Health status information
        """
        response = self._make_request("GET", "/query/health")
        return response.json()

    def query_documents(
        self,
        question: str,
        equipment: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query documents with a question.

        Args:
            question: User's question
            equipment: Equipment filter (optional)
            chat_history: Previous chat messages (optional)
            request_id: Optional request identifier required by backend.
                If not provided, generated.

        Returns:
            Query response with answer and sources
        """
        # Ensure we always send request_id as backend requires it
        rid = request_id or str(uuid.uuid4())
        payload = {
            "request_id": rid,
            "question": question,
            "equipment": equipment,
            "chat_history": chat_history or [],
        }

        response = self._make_request("POST", "/query/query", json=payload)
        return response.json()

    def reindex_documents(self) -> Dict[str, Any]:
        """
        Trigger document reindexing.

        Returns:
            Reindex operation result
        """
        response = self._make_request("POST", "/query/reindex", timeout=180)
        return response.json()

    def get_index_size(self) -> Optional[int]:
        """
        Get current index size from health check.

        Returns:
            Number of indexed document chunks, or None if unavailable
        """
        try:
            health_data = self.health_check()
            vector_store_info = health_data.get("components", {}).get(
                "vector_store", ""
            )

            # Parse index size from string representation
            import re

            match = re.search(r"'index_size': (\d+)", str(vector_store_info))
            if match:
                return int(match.group(1))
            return None
        except Exception as e:
            logger.error(f"Failed to get index size: {e}")
            return None


@st.cache_resource
def get_api_client(base_url: str) -> MedRAGAPIClient:
    """
    Get cached API client instance.

    Args:
        base_url: Base URL for the API

    Returns:
        API client instance
    """
    return MedRAGAPIClient(base_url)

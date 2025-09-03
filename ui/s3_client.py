import logging
import os
from typing import Any, Dict, List

import streamlit as st
from minio import Minio

logger = logging.getLogger(__name__)


class S3DocumentClient:
    """Client for S3 document operations."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket_name: str,
        secure: bool = False,
        region: str = "us-east-1",
    ):
        """
        Initialize S3 client.

        Args:
            endpoint: S3 endpoint
            access_key: S3 access key
            secret_key: S3 secret key
            bucket_name: S3 bucket name
            secure: Whether to use HTTPS
            region: S3 region
        """
        self.bucket_name = bucket_name
        self.client = None
        self.endpoints_to_try = [
            (endpoint, "Primary endpoint"),
            ("localhost:9000", "Local development fallback"),
        ]
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure
        self.region = region

    def _get_client(self) -> Minio:
        """
        Get MinIO client with connection fallback.

        Returns:
            Connected MinIO client

        Raises:
            Exception: If no connection can be established
        """
        if self.client:
            return self.client

        for endpoint, desc in self.endpoints_to_try:
            try:
                client = Minio(
                    endpoint=endpoint,
                    access_key=self.access_key,
                    secret_key=self.secret_key,
                    secure=self.secure,
                    region=self.region,
                )
                # Test connection
                client.list_buckets()
                self.client = client
                logger.info(f"Connected to S3 via {desc}: {endpoint}")
                return client
            except Exception as e:
                logger.debug(f"Failed to connect via {desc} ({endpoint}): {e}")
                continue

        raise Exception("Could not connect to any S3 endpoint")

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all PDF documents in the bucket.

        Returns:
            List of document information dictionaries
        """
        try:
            client = self._get_client()
            objects = client.list_objects(self.bucket_name, recursive=True)
            documents = []

            for obj in objects:
                if obj.object_name.lower().endswith(".pdf"):
                    # Extract equipment from path
                    parts = obj.object_name.split("/")
                    equipment = parts[0] if len(parts) > 1 else "general"
                    filename = parts[-1]

                    documents.append(
                        {
                            "key": obj.object_name,
                            "filename": filename,
                            "equipment": equipment,
                            "size": obj.size,
                            "last_modified": (
                                obj.last_modified.strftime("%Y-%m-%d %H:%M:%S")
                                if obj.last_modified
                                else "Unknown"
                            ),
                        }
                    )

            return sorted(documents, key=lambda x: (x["equipment"], x["filename"]))

        except Exception as e:
            logger.error(f"Failed to list S3 documents: {e}")
            return []

    def get_equipment_list(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Get unique equipment list from documents.

        Args:
            documents: List of document dictionaries

        Returns:
            Sorted list of unique equipment names
        """
        equipment_set = set(doc["equipment"] for doc in documents)
        return sorted(equipment_set)

    def filter_documents_by_equipment(
        self, documents: List[Dict[str, Any]], equipment: str
    ) -> List[Dict[str, Any]]:
        """
        Filter documents by equipment.

        Args:
            documents: List of document dictionaries
            equipment: Equipment name to filter by

        Returns:
            Filtered list of documents
        """
        return [doc for doc in documents if doc["equipment"] == equipment]


@st.cache_resource
def get_s3_client() -> S3DocumentClient:
    """
    Get cached S3 client instance.

    Returns:
        S3 client instance
    """
    endpoint = os.getenv("S3_ENDPOINT", "localhost:9000")
    access_key = os.getenv("S3_ACCESS_KEY", "minioadmin")
    secret_key = os.getenv("S3_SECRET_KEY", "minioadmin123")
    bucket_name = os.getenv("S3_BUCKET_NAME", "medical-documents")
    region = os.getenv("S3_REGION", "us-east-1")
    secure_env = os.getenv("S3_SECURE", "false")
    secure = str(secure_env).strip().lower() in {"1", "true", "yes", "y"}

    return S3DocumentClient(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        bucket_name=bucket_name,
        secure=secure,
        region=region,
    )

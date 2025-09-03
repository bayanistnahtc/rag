from enum import Enum
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class DocumentSource(str, Enum):
    """Enum for document source types."""
    LOCAL = "local"
    S3 = "s3"


class Settings(BaseSettings):
    PROJECT_NAME: str = "Rag Medical Assistant"
    API_V1_STR: str = "/api/v1"
    
    # Server settings
    host: str = Field(default="127.0.0.1", env="HOST")
    port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")

    # Document source configuration
    document_source: DocumentSource = Field(default=DocumentSource.LOCAL, env="DOCUMENT_SOURCE")
    
    # S3/MinIO configuration
    s3_endpoint: str = Field(default="localhost:9000", env="S3_ENDPOINT")
    s3_access_key: str = Field(default="minioadmin", env="S3_ACCESS_KEY")
    s3_secret_key: str = Field(default="minioadmin123", env="S3_SECRET_KEY")
    s3_bucket_name: str = Field(default="medical-documents", env="S3_BUCKET_NAME")
    s3_secure: bool = Field(default=False, env="S3_SECURE")  # False for local MinIO
    s3_region: str = Field(default="us-east-1", env="S3_REGION")

    class Config:
        env_file = ".env"
        extra = "ignore"

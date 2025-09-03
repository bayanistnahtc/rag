import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, cast

from dotenv import load_dotenv

load_dotenv()


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""

    model_name: str = "intfloat/multilingual-e5-small"
    model_kwargs: Dict[str, Any] = field(default_factory=lambda: {"device": "cpu"})
    encode_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"normalize_embeddings": True}
    )

    def __post_init__(self):
        if self.model_kwargs is None:
            self.model_kwargs = {"device": "cpu"}
        if self.encode_kwargs is None:
            self.encode_kwargs = {"normalize_embeddings": True}

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """Create EmbeddingConfig from environment variables."""
        model_name = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small")
        model_device = os.getenv("EMBEDDING_DEVICE", "cpu")
        normalize_embeddings = (
            os.getenv("EMBEDDING_NORMALIZE", "true").lower() == "true"
        )
        return cls(
            model_name=model_name,
            model_kwargs={"device": model_device},
            encode_kwargs={"normalize_embeddings": normalize_embeddings},
        )


@dataclass
class VectorStoreConfig:
    """Configuration for vector store operations."""

    index_path: Path = Path("data/indexes")
    chunk_size: int = 1000
    chunk_overlap: int = 200

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

    @classmethod
    def from_env(cls) -> "VectorStoreConfig":
        """Create VectorStoreConfig from environment variables."""
        index_path = Path(os.getenv("VECTOR_INDEX_PATH", "data/indexes"))
        chunk_size = int(os.getenv("VECTOR_CHUNK_SIZE", "1000"))
        chunk_overlap = int(os.getenv("VECTOR_CHUNK_OVERLAP", "200"))
        return cls(
            index_path=index_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


@dataclass
class RetrievalConfig:
    """Configuration for document retrieval."""

    k: int = 10
    bm25_weight: float = 0.5
    faiss_weight: float = 0.5
    rerank_top_k: int = 100

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.k <= 0:
            raise ValueError("k must be positive")
        if not (0 <= self.bm25_weight <= 1):
            raise ValueError("bm25_weight must be between 0 and 1")
        if not (0 <= self.faiss_weight <= 1):
            raise ValueError("faiss_weight must be between 0 and 1")
        if abs(self.bm25_weight + self.faiss_weight - 1.0) > 1e-6:
            raise ValueError("bm25_weight + faiss_weight must equal 1.0")
        if self.rerank_top_k <= 0:
            raise ValueError("rerank_top_k must be positive")

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        """Create RetrievalConfig from environment variables."""
        k = int(os.getenv("RETRIEVAL_K", "10"))
        bm25_weight = float(os.getenv("RETRIEVAL_BM25_WEIGHT", "0.5"))
        faiss_weight = float(os.getenv("RETRIEVAL_FAISS_WEIGHT", "0.5"))
        rerank_top_k = int(os.getenv("RETRIEVAL_RERANK_TOP_K", "100"))
        return cls(
            k=k,
            bm25_weight=bm25_weight,
            faiss_weight=faiss_weight,
            rerank_top_k=rerank_top_k,
        )


@dataclass
class LLMConfig:
    """Configuration for Large Language Model."""

    model_type: str
    model_name: str
    model_url: str
    model_api_key: str
    max_tokens: int = 2048
    temperature: float = 0.1
    timeout: int = 30

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not all([self.model_type, self.model_name, self.model_url]):
            raise ValueError("model_type, model_name, and model_url are required")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not (0 <= self.temperature <= 2):
            raise ValueError("temperature must be between 0 and 2")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create LLMConfig from environment variables."""
        return cls(
            model_type=os.getenv("LLM_MODEL_TYPE", ""),
            model_name=os.getenv("LLM_MODEL_NAME", ""),
            model_url=os.getenv("LLM_MODEL_URL", "localhost"),
            model_api_key=os.getenv("LLM_MODEL_API_KEY", ""),
            max_tokens=int(os.getenv("MAX_TOKENS", "2048")),
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
            timeout=int(os.getenv("LLM_TIMEOUT", "30")),
        )


@dataclass
class RecursiveCharacterSplitterConfig:
    """Configuration for recursive character text splitter."""

    type: Literal["recursive_character"] = "recursive_character"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # TODO: add Chunking validations

    @classmethod
    def from_env(cls) -> "RecursiveCharacterSplitterConfig":
        """Create RecursiveCharacterSplitterConfig from environment variables."""
        chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        return cls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


@dataclass
class SemanticSplitterConfig:
    """Configuration for semantic text splitter."""

    type: Literal["semantic"] = "semantic"
    model_name: str = "intfloat/multilingual-e5-small"
    breakpoint_threshold_type: Literal[
        "percentile", "standard_deviation", "interquartile"
    ] = "percentile"
    breakpoint_threshold_amount: float = 0.95
    window_size: int = 5
    overlap: int = 1
    min_chunk_words: int = 50

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not (0 < self.breakpoint_threshold_amount <= 1):
            raise ValueError("breakpoint_threshold_amount must be between 0 and 1")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.overlap < 0:
            raise ValueError("overlap cannot be negative")
        if self.min_chunk_words <= 0:
            raise ValueError("min_chunk_words must be positive")

    @classmethod
    def from_env(cls) -> "SemanticSplitterConfig":
        """Create SemanticSplitterConfig from environment variables."""
        model_name = os.getenv("SEMANTIC_MODEL_NAME", "intfloat/multilingual-e5-small")
        breakpoint_threshold_type_raw = os.getenv(
            "BREAKPOINT_THRESHOLD_TYPE", "percentile"
        )
        # Validate and cast to Literal type
        valid_types = ("percentile", "standard_deviation", "interquartile")
        if breakpoint_threshold_type_raw not in valid_types:
            raise ValueError(f"breakpoint_threshold_type must be one of {valid_types}")
        breakpoint_threshold_type: Literal[
            "percentile", "standard_deviation", "interquartile"
        ] = cast(
            Literal["percentile", "standard_deviation", "interquartile"],
            breakpoint_threshold_type_raw,
        )
        breakpoint_threshold_amount = float(
            os.getenv("BREAKPOINT_THRESHOLD_AMOUNT", "0.95")
        )
        window_size = int(os.getenv("SEMANTIC_WINDOW_SIZE", "5"))
        overlap = int(os.getenv("SEMANTIC_OVERLAP", "1"))
        min_chunk_words = int(os.getenv("SEMANTIC_MIN_CHUNK_WORDS", "50"))
        return cls(
            model_name=model_name,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            window_size=window_size,
            overlap=overlap,
            min_chunk_words=min_chunk_words,
        )


@dataclass
class PageBasedSplitterConfig:
    """Configuration for page-based text splitter."""

    type: Literal["page_based"] = "page_based"
    min_chunk_words: int = 10

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.min_chunk_words <= 0:
            raise ValueError("min_chunk_words must be positive")

    @classmethod
    def from_env(cls) -> "PageBasedSplitterConfig":
        """Create PageBasedSplitterConfig from environment variables."""
        min_chunk_words = int(os.getenv("PAGE_BASED_MIN_CHUNK_WORDS", "10"))
        return cls(min_chunk_words=min_chunk_words)


@dataclass
class ParagraphSplitterConfig:
    type: Literal["paragraph"] = "paragraph"
    min_chunk_words: int = 10
    max_chunk_words: int = 300

    @classmethod
    def from_env(cls) -> "ParagraphSplitterConfig":
        return cls(
            min_chunk_words=int(os.getenv("PARAGRAPH_MIN_CHUNK_WORDS", "10")),
            max_chunk_words=int(os.getenv("PARAGRAPH_MAX_CHUNK_WORDS", "300")),
        )


@dataclass
class DocumentsConfig:
    """Configuration for loading and processing document files (e.g., PDFs)."""

    # Path to a single file or directory to scan (for local files)
    # For S3, this can be used as a prefix filter
    file_path: str

    # Which loader to use: PyMuPDF, pdfminer.six, or Unstructured
    loader_type: str = "PyMuPDF"

    # Which file extensions to ingest
    file_extensions: List[str] = field(default_factory=lambda: [".pdf"])

    # PDF-specific options
    password: Optional[str] = None
    pages: Optional[List[int]] = None  # e.g. [1,2,5] or None for all

    # Any extra metadata fields to attach to each chunk
    metadata: Dict[str, str] = field(default_factory=dict)

    # Timeout (in seconds) for any I/O operations
    timeout: int = 30

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # For S3 sources, file_path can be empty (used as prefix filter)
        # For local sources, file_path is required
        # Note: This validation is relaxed to support S3 usage

        # File extensions
        if not self.file_extensions:
            raise ValueError("file_extensions must not be empty")

        # Timeout
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

        # Loader type
        supported = {"PyMuPDF", "pdfminer", "unstructured"}
        if self.loader_type not in supported:
            raise ValueError(
                f"Unsupported loader_type '{self.loader_type}'. "
                f"Choose one of {supported}"
            )

        # Pages list
        if self.pages is not None:
            if not all(isinstance(p, int) and p > 0 for p in self.pages):
                raise ValueError("pages must be a list of positive integers")

    @classmethod
    def from_env(cls) -> "DocumentsConfig":
        """Create DocumentsConfig from environment variables."""

        # Helper to parse comma-separated ints
        def _parse_int_list(env_val: Optional[str]) -> Optional[List[int]]:
            if not env_val:
                return None
            nums = [s.strip() for s in env_val.split(",") if s.strip()]
            return [int(n) for n in nums]

        return cls(
            file_path=os.getenv("DOC_FILE_PATH", ""),
            loader_type=os.getenv("DOC_LOADER_TYPE", "PyMuPDF"),
            file_extensions=os.getenv("DOC_EXTENSIONS", ".pdf").split(","),
            password=os.getenv("DOC_PASSWORD") or None,
            pages=_parse_int_list(os.getenv("DOC_PAGES")),
            timeout=int(os.getenv("DOC_TIMEOUT", "30")),
        )


@dataclass
class CompressionConfig:
    """Configuration for contextual compression / lightweight reranking."""

    # Embeddings used by EmbeddingsFilter within ContextualCompressionRetriever
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs: Dict[str, Any] = field(default_factory=lambda: {"device": "cpu"})
    encode_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"prompt": "passage:", "normalize_embeddings": True}
    )
    query_encode_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"prompt": "query:", "normalize_embeddings": True}
    )

    # EmbeddingsFilter params
    similarity_threshold: float = 0.20
    top_k: int = 80

    @classmethod
    def from_env(cls) -> "CompressionConfig":
        def _json_load(env_name: str, default_json: str) -> Dict[str, Any]:
            try:
                return json.loads(os.getenv(env_name, default_json))
            except json.JSONDecodeError:
                return json.loads(default_json)

        return cls(
            model_name=os.getenv(
                "RERANKER_EMBEDDINGS_MODEL_NAME",
                "intfloat/multilingual-e5-small",
            ),
            model_kwargs=_json_load(
                "RERANKER_EMBEDDINGS_MODEL_KWARGS", '{"device": "cpu"}'
            ),
            encode_kwargs=_json_load(
                "RERANKER_EMBEDDINGS_ENCODE_KWARGS",
                '{"prompt": "passage:", "normalize_embeddings": true}',
            ),
            query_encode_kwargs=_json_load(
                "RERANKER_EMBEDDINGS_QUERY_ENCODE_KWARGS",
                '{"prompt": "query:", "normalize_embeddings": true}',
            ),
            similarity_threshold=float(
                os.getenv("COMPRESSION_SIMILARITY_THRESHOLD", "0.20")
            ),
            top_k=int(os.getenv("COMPRESSION_TOP_K", "80")),
        )


@dataclass
class ChatHistoryConfig:
    """Chat history management configurations"""

    max_token_limit: int = 5000
    memory_type: str = "token_buffer"  # "token_buffer", "window", "summary_buffer"
    window_size: int = 10  # QA pairs
    session_ttl: int = 3600

    @classmethod
    def from_env(cls) -> "ChatHistoryConfig":
        """Create configuration from environment variables"""
        return cls(
            max_token_limit=int(os.getenv("CHAT_HISTORY_MAX_TOKENS", "5000")),
            memory_type=os.getenv("CHAT_HISTORY_MEMORY_TYPE", "token_buffer"),
            window_size=int(os.getenv("CHAT_HISTORY_WINDOW_SIZE", "10")),
            session_ttl=int(os.getenv("CHAT_HISTORY_SESSION_TTL", "3600")),
        )


@dataclass
class RAGConfig:
    """Complete RAG system configurations"""

    documents_config: DocumentsConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    text_splitter: Union[
        SemanticSplitterConfig,
        RecursiveCharacterSplitterConfig,
        PageBasedSplitterConfig,
        ParagraphSplitterConfig,
    ]
    # Accepts either: RecursiveCharacterSplitterConfig,
    # SemanticSplitterConfig, PageBasedSplitterConfig, or ParagraphSplitterConfig
    retrieval: RetrievalConfig
    compression: CompressionConfig
    chat_history: ChatHistoryConfig
    llm: LLMConfig

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Create RAGConfig from environment variables with defaults."""
        documents_config = DocumentsConfig.from_env()
        splitter_type = os.getenv(
            "CHUNKER_TYPE", "paragraph"
        )  # Changed default to page_based
        text_splitter: Union[
            RecursiveCharacterSplitterConfig,
            SemanticSplitterConfig,
            PageBasedSplitterConfig,
            ParagraphSplitterConfig,
        ]
        if splitter_type == "semantic":
            text_splitter = SemanticSplitterConfig.from_env()
        elif splitter_type == "page_based":
            text_splitter = PageBasedSplitterConfig.from_env()
        elif splitter_type == "paragraph":
            text_splitter = ParagraphSplitterConfig.from_env()
        else:
            text_splitter = RecursiveCharacterSplitterConfig.from_env()
        return cls(
            compression=CompressionConfig.from_env(),
            documents_config=documents_config,
            embedding=EmbeddingConfig.from_env(),
            vector_store=VectorStoreConfig.from_env(),
            text_splitter=text_splitter,
            retrieval=RetrievalConfig.from_env(),
            chat_history=ChatHistoryConfig.from_env(),
            llm=LLMConfig.from_env(),
        )

import logging
import re
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from data_processing.base import BaseTextSplitter

logger = logging.getLogger(__name__)


def preprocess_text(text: str) -> str:
    """
    Clean hyphenation at line breaks and normalize line breaks within paragraphs.
    """
    # Remove soft hyphens and join hyphenated line breaks
    text = re.sub(r"(\w+)[\-­]\n(\w+)", r"\1\2", text)
    # Remove soft hyphen unicode (U+00AD)
    text = text.replace("\u00ad", "")
    # Replace single newlines within paragraphs with space
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Normalize multiple spaces
    text = re.sub(r" +", " ", text)
    return text


class RecursiveCharacterTextSplitterWrapper(BaseTextSplitter):
    """
    Wrapper for RecursiveCharacterTextSplitter from LangChain.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        logger.info(f"Splitting {len(documents)} documents...")
        # Preprocess text for each document
        preprocessed_docs = [
            Document(
                page_content=preprocess_text(doc.page_content), metadata=doc.metadata
            )
            for doc in documents
        ]
        return self._splitter.split_documents(preprocessed_docs)


class SemanticTextSplitter(BaseTextSplitter):
    """
    Section-based: Splits by detected headings/sections,
    merges all text under a heading into a chunk, applies overlap,
    and ensures minimum chunk size.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 0.95,
        window_size: int = 5,
        overlap: int = 1,
        min_chunk_words: int = 50,
    ):
        self.model = SentenceTransformer(model_name)
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.window_size = window_size
        self.overlap = overlap
        self.min_chunk_words = min_chunk_words

    def _is_heading(self, line):
        # e.g., '3.6.8 ...', '2.1 ...', '1 ...', or all-caps
        return bool(re.match(r"^\d+(\.\d+)*\s+", line.strip()))

    def _is_list_item(self, sentence):
        return bool(re.match(r"^\s*(\d+\.|[-•*])", sentence.strip()))

    def _is_section_title(self, sentence):
        s = sentence.strip()
        return len(s.split()) <= 8 and s.istitle() and not self._is_list_item(s)

    def _is_reference_or_pagenum(self, line):
        s = line.strip()
        if not s:
            return True
        if len(s) < 4:
            return True
        if re.match(r"^\d+$", s):
            return True
        if re.search(r"Vol\\.|P\\.|\d+\(\d+\)|\d+-\d+", s):
            return True
        if re.match(r"^—", s):
            return True
        return False

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks = []
        logger.info(f"Starting semantic splitting of {len(documents)} documents")

        for doc_idx, doc in enumerate(documents):
            logger.debug(
                f"Processing document {doc_idx + 1}/{len(documents)}: "
                f"{doc.metadata.get('file_name', 'unknown')} "
                f"page {doc.metadata.get('page_number', 'unknown')}"
            )

            # Preprocess text
            clean_content = preprocess_text(doc.page_content)
            logger.debug(
                f"Original content length: {len(doc.page_content)}, "
                f"Cleaned content length: {len(clean_content)}"
            )

            # Pre-filter lines
            lines = clean_content.splitlines()
            filtered_lines = [
                line for line in lines if not self._is_reference_or_pagenum(line)
            ]
            logger.debug(
                f"Original lines: {len(lines)}, Filtered lines: {len(filtered_lines)}"
            )

            # Section-based chunking
            section_chunks: List[tuple[Optional[str], str]] = []
            current_section: List[str] = []
            current_title: Optional[str] = None

            for line in filtered_lines:
                if self._is_heading(line):
                    # Save previous section
                    if current_section:
                        section_text = "\n".join(current_section)
                        section_chunks.append((current_title, section_text))
                        logger.debug(
                            f"Added section: '{current_title}' with "
                            f"{len(section_text.split())} words"
                        )
                    current_title = line.strip()
                    current_section = []
                else:
                    current_section.append(line)

            # Add last section
            if current_section:
                section_text = "\n".join(current_section)
                section_chunks.append((current_title, section_text))
                logger.debug(
                    f"Added final section: '{current_title}' with "
                    f"{len(section_text.split())} words"
                )

            logger.debug(f"Found {len(section_chunks)} sections in document")

            # Merge small sections with next (but be less aggressive)
            merged_chunks = []
            i = 0
            while i < len(section_chunks):
                title, text = section_chunks[i]
                words = text.split()

                # Only merge if section is very small (reduce threshold)
                merge_threshold = max(
                    10, self.min_chunk_words // 2
                )  # More conservative merging

                # If too small, merge with next (but limit merging to avoid huge chunks)
                merge_count = 0
                while (
                    len(words) < merge_threshold
                    and i + 1 < len(section_chunks)
                    and merge_count < 2
                ):
                    next_title, next_text = section_chunks[i + 1]
                    text = text + "\n" + next_text
                    words = text.split()
                    i += 1
                    merge_count += 1
                    logger.debug(
                        f"Merged section {i} with next, total words: {len(words)}"
                    )

                merged_chunks.append((title, text))
                i += 1

            logger.debug(f"After merging: {len(merged_chunks)} chunks")

            # Apply windowed overlap (but be more conservative)
            for j in range(len(merged_chunks)):
                start = max(0, j - self.overlap)
                end = min(len(merged_chunks), j + 1 + self.overlap)
                window_text = "\n".join(
                    [merged_chunks[k][1] for k in range(start, end)]
                )
                window_title = merged_chunks[j][0]

                # More lenient minimum word count
                min_words = max(5, self.min_chunk_words // 4)
                if len(window_text.split()) >= min_words:
                    meta = dict(doc.metadata)
                    if window_title:
                        meta["section_title"] = window_title

                    chunk = Document(page_content=window_text.strip(), metadata=meta)
                    chunks.append(chunk)
                    logger.debug(
                        f"Created chunk {len(chunks)}: '{window_title}' with "
                        f"{len(window_text.split())} words"
                    )
                else:
                    logger.debug(
                        f"Skipped chunk '{window_title}' - too small "
                        f"({len(window_text.split())} words)"
                    )

        logger.info(
            f"Semantic splitting complete: {len(documents)} documents -> "
            f"{len(chunks)} chunks"
        )
        return chunks


class PageBasedTextSplitter(BaseTextSplitter):
    """
    Simple page-based splitter that preserves page boundaries.
    Useful when you want to maintain page-level granularity for retrieval.
    """

    def __init__(self, min_chunk_words: int = 10):
        self.min_chunk_words = min_chunk_words

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks = []
        logger.info(f"Starting page-based splitting of {len(documents)} documents")

        for doc_idx, doc in enumerate(documents):
            logger.debug(
                f"Processing document {doc_idx + 1}/{len(documents)}: "
                f"{doc.metadata.get('file_name', 'unknown')} "
                f"page {doc.metadata.get('page_number', 'unknown')}"
            )

            # Preprocess text
            clean_content = preprocess_text(doc.page_content)

            # Split by paragraphs or sections within the page
            paragraphs = [p.strip() for p in clean_content.split("\n\n") if p.strip()]

            if not paragraphs:
                # If no paragraphs, split by lines
                paragraphs = [
                    line.strip() for line in clean_content.splitlines() if line.strip()
                ]

            # Create chunks from paragraphs
            current_chunk: List[str] = []
            current_words = 0

            for paragraph in paragraphs:
                paragraph_words = len(paragraph.split())

                # If adding this paragraph would make chunk too large,
                # save current chunk
                if current_words + paragraph_words > 1000 and current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    if len(chunk_text.split()) >= self.min_chunk_words:
                        meta = dict(doc.metadata)
                        meta["chunk_type"] = "page_paragraph"
                        chunk = Document(page_content=chunk_text, metadata=meta)
                        chunks.append(chunk)
                        logger.debug(
                            f"Created paragraph chunk with {len(chunk_text.split())} "
                            "words"
                        )

                    current_chunk = [paragraph]
                    current_words = paragraph_words
                else:
                    current_chunk.append(paragraph)
                    current_words += paragraph_words

            # Add remaining content as final chunk
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                if len(chunk_text.split()) >= self.min_chunk_words:
                    meta = dict(doc.metadata)
                    meta["chunk_type"] = "page_paragraph"
                    chunk = Document(page_content=chunk_text, metadata=meta)
                    chunks.append(chunk)
                    logger.debug(
                        f"Created final paragraph chunk with {len(chunk_text.split())} "
                        "words"
                    )

            # If no chunks were created (content too small), create a single chunk
            if (
                not chunks
                or chunks[-1].metadata.get("file_name") != doc.metadata.get("file_name")
                or chunks[-1].metadata.get("page_number")
                != doc.metadata.get("page_number")
            ):
                if len(clean_content.split()) >= self.min_chunk_words:
                    meta = dict(doc.metadata)
                    meta["chunk_type"] = "page_full"
                    chunk = Document(page_content=clean_content, metadata=meta)
                    chunks.append(chunk)
                    logger.debug(
                        f"Created full page chunk with {len(clean_content.split())} "
                        "words"
                    )

        logger.info(
            f"Page-based splitting complete: {len(documents)} documents -> "
            f"{len(chunks)} chunks"
        )
        return chunks


class ParagraphTextSplitter(BaseTextSplitter):
    """
    Splits documents by paragraphs (double newlines).
    Optionally merges short paragraphs.
    """

    def __init__(self, min_chunk_words: int = 10, max_chunk_words: int = 300):
        self.min_chunk_words = min_chunk_words
        self.max_chunk_words = max_chunk_words

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks = []
        for doc in documents:
            paragraphs = [
                p.strip()
                for p in preprocess_text(doc.page_content).split("\n\n")
                if p.strip()
            ]
            current_chunk: List[str] = []
            current_words = 0
            for para in paragraphs:
                para_words = len(para.split())
                if current_words + para_words > self.max_chunk_words and current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    if len(chunk_text.split()) >= self.min_chunk_words:
                        meta = dict(doc.metadata)
                        meta["chunk_type"] = "paragraph"
                        chunks.append(Document(page_content=chunk_text, metadata=meta))
                    current_chunk = [para]
                    current_words = para_words
                else:
                    current_chunk.append(para)
                    current_words += para_words
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                if len(chunk_text.split()) >= self.min_chunk_words:
                    meta = dict(doc.metadata)
                    meta["chunk_type"] = "paragraph"
                    chunks.append(Document(page_content=chunk_text, metadata=meta))
        return chunks


class TextSplitterFactory:
    """
    Factory for creating text splitter instances.
    """

    @staticmethod
    def create(config):
        if getattr(config, "type", None) == "recursive_character":
            return RecursiveCharacterTextSplitterWrapper(
                chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
            )
        elif getattr(config, "type", None) == "semantic":
            return SemanticTextSplitter(
                model_name=config.model_name,
                breakpoint_threshold_type=config.breakpoint_threshold_type,
                breakpoint_threshold_amount=config.breakpoint_threshold_amount,
                window_size=config.window_size,
                overlap=config.overlap,
                min_chunk_words=config.min_chunk_words,
            )
        elif getattr(config, "type", None) == "page_based":
            return PageBasedTextSplitter(
                min_chunk_words=getattr(config, "min_chunk_words", 10)
            )
        elif getattr(config, "type", None) == "paragraph":
            return ParagraphTextSplitter(
                min_chunk_words=getattr(config, "min_chunk_words", 10),
                max_chunk_words=getattr(config, "max_chunk_words", 300),
            )
        raise ValueError(f"Unknown splitter type: {getattr(config, 'type', None)}")

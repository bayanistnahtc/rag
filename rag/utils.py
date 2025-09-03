from typing import Any, Dict, List

from langchain_core.documents import Document

from rag.models import Answer


def format_documents(docs: List[Document]) -> str:
    """
    Format retrieved documents for context injection.

    Args:
        docs (List[Document]): list of Document instances retrieved by the vector store.

    Returns: str
        A formatted string with each document chunk prefixed by its source tag.
    """

    if not docs:
        return "No relevant context found."

    formatted: List[str] = []
    for i, doc in enumerate(docs):
        file_path = doc.metadata.get("file_name", "UNKNOWN")
        page_num = doc.metadata.get("page_number", "UNKNOWN")  # Use page_number field
        if page_num == "UNKNOWN":
            page_num = doc.metadata.get("page", "UNKNOWN")
        if page_num == "UNKNOWN":
            page_num = doc.metadata.get("page_label", "UNKNOWN")

        # Extract just the filename from the full path
        file_name = (
            file_path.split("\\")[-1]
            if "\\" in file_path
            else file_path.split("/")[-1] if "/" in file_path else file_path
        )

        source_tag = f"[{file_name}/PAGE_{page_num}]"
        formatted.append(f"Source {i+1} {source_tag}:\n{doc.page_content}")
    return "\n\n".join(formatted)


def docs_to_answer(data: Dict[str, Any]) -> Answer:
    """
    Transform chain output to Answer model.

    Args:
        data (Dict[str, Any]): The dictionary expected to have keys
        'answer_text', 'retrieved_docs', and optionally 'needs_clarification'.

    Returns: Answer
        An Answer instance encapsulating the LLM response and its sources.
    """
    # Process source chunks to ensure they have proper metadata
    source_chunks = data["retrieved_docs"]
    for chunk in source_chunks:
        # Ensure page number is properly set
        if "page_number" not in chunk.metadata:
            if "page" in chunk.metadata:
                chunk.metadata["page_number"] = chunk.metadata["page"]
            elif "page_label" in chunk.metadata:
                chunk.metadata["page_number"] = chunk.metadata["page_label"]

        # Extract filename from file_path if needed
        if "file_path" in chunk.metadata and "file_name" not in chunk.metadata:
            file_path = chunk.metadata["file_path"]
            file_name = (
                file_path.split("\\")[-1]
                if "\\" in file_path
                else file_path.split("/")[-1] if "/" in file_path else file_path
            )
            chunk.metadata["file_name"] = file_name

    return Answer(
        answer_text=data["answer_text"] if data["answer_text"] else "No answer",
        source_chunks=source_chunks,
        response_type=data.get("response_type", "ANSWER"),
    )

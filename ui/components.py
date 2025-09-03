import logging
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

from ui.api_client import MedRAGAPIClient
from ui.s3_client import S3DocumentClient

logger = logging.getLogger(__name__)


def render_document_source_selector() -> str:
    """
    Render document source selection widget.

    Returns:
        Selected document source ("s3" or "local")
    """
    return st.selectbox(
        "Document Source",
        options=["s3", "local"],
        index=0,  # S3 by default
        help="Select whether to use S3/MinIO storage or local files",
    )


def render_local_document_selector() -> str:
    """
    Render local document selection interface.

    Returns:
        Selected document path or empty string
    """
    import os

    pdf_dir = os.path.join("data", "pdfs")
    if os.path.exists(pdf_dir):
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
        pdf_options = [os.path.join(pdf_dir, f) for f in pdf_files]
        pdf_display = [f for f in pdf_files]
    else:
        pdf_files = []
        pdf_options = []
        pdf_display = []

    if pdf_options:
        selected_pdf_idx = st.selectbox(
            "Select Document",
            options=range(len(pdf_options)),
            format_func=lambda i: pdf_display[i],
            help="Select the PDF document to filter answers by source",
            index=0,
        )
        return pdf_options[selected_pdf_idx]
    else:
        st.warning("No PDF files found in local data/pdfs directory")
        return ""


def render_s3_document_browser(
    s3_client: S3DocumentClient,
) -> Tuple[str, Optional[str]]:
    """
    Render S3 document browser interface.

    Args:
        s3_client: S3 client instance

    Returns:
        Tuple of (selected_equipment, selected_document_key)
    """
    st.subheader("📁 S3 Documents")

    # Get documents from S3
    with st.spinner("Loading documents from S3..."):
        documents = s3_client.list_documents()

    if not documents:
        st.warning("No documents found in S3 bucket")
        return "", None

    # Document statistics
    equipment_list = s3_client.get_equipment_list(documents)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", len(documents))
    with col2:
        st.metric("Equipment Types", len(equipment_list))
    with col3:
        if st.button("🔄 Refresh Documents"):
            st.cache_data.clear()
            st.rerun()

    # Equipment filter
    equipment_options = ["All Equipment"] + equipment_list
    selected_equipment = st.selectbox(
        "Filter by Equipment",
        options=equipment_options,
        index=0,
        help="Select equipment to filter documents",
    )

    # Filter documents
    if selected_equipment != "All Equipment":
        filtered_docs = s3_client.filter_documents_by_equipment(
            documents, selected_equipment
        )
        equipment_filter = selected_equipment
    else:
        filtered_docs = documents
        equipment_filter = ""

    # Document selection
    selected_document = None
    if filtered_docs:
        st.write(f"**Documents ({len(filtered_docs)}):**")

        # Document selection dropdown
        doc_options = ["All Documents"] + [
            f"{doc['equipment']}/{doc['filename']}" for doc in filtered_docs
        ]
        selected_doc_display = st.selectbox(
            "Select Specific Document (optional)",
            options=doc_options,
            index=0,
            help="Select a specific document to search within, "
            "or 'All Documents' to search all",
        )

        if selected_doc_display != "All Documents":
            # Find the selected document
            for doc in filtered_docs:
                if f"{doc['equipment']}/{doc['filename']}" == selected_doc_display:
                    # Use the full S3 key (equipment/filename)
                    # as document identifier for proper filtering
                    selected_document = doc["key"]  # This contains the full S3 path
                    # Set equipment_filter to the document's equipment
                    # for backend filtering
                    equipment_filter = doc["equipment"]
                    break

        # Create DataFrame for display
        df_data = []
        for doc in filtered_docs:
            is_selected = selected_document == doc["key"]
            df_data.append(
                {
                    "Selected": "✅" if is_selected else "",
                    "Equipment": doc["equipment"],
                    "Filename": doc["filename"],
                    "Size (KB)": (
                        f"{doc['size'] / 1024:.1f}" if doc["size"] else "Unknown"
                    ),
                    "Last Modified": doc["last_modified"],
                }
            )

        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Show selection info
        if selected_document:
            st.info(f"🎯 Searching within: **{selected_doc_display}**")
        elif equipment_filter:
            st.info(f"🔧 Searching within equipment: **{equipment_filter}**")
        else:
            st.info("📁 Searching all documents in S3 bucket")
    else:
        st.warning("No documents found for selected equipment")

    return equipment_filter, selected_document


def render_document_management(api_client: MedRAGAPIClient) -> None:
    """
    Render document management interface.

    Args:
        api_client: API client instance
    """
    st.markdown("---")
    st.subheader("🔄 Document Management")

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "🔄 Reindex Documents", help="Rebuild search index with all documents"
        ):
            with st.spinner("Reindexing documents... This may take a few minutes."):
                try:
                    result = api_client.reindex_documents()
                    st.success("✅ Reindexing completed!")
                    st.info(f"📊 Indexed {result.get('index_size', 0)} document chunks")
                    st.info(
                        f"⏱️ Processing time: {result.get('processing_time_ms', 0)}ms"
                    )
                    # Clear caches to refresh
                    st.cache_data.clear()
                    st.cache_resource.clear()
                except Exception as e:
                    st.error(f"❌ Reindexing error: {e}")

    with col2:
        if st.button("📊 Check Index Status", help="Check current index statistics"):
            try:
                index_size = api_client.get_index_size()
                if index_size is not None:
                    st.success(f"📊 Current index size: {index_size} document chunks")
                else:
                    st.warning("📊 Could not retrieve index size")
            except Exception as e:
                st.error(f"❌ Status check error: {e}")


def render_chat_interface(
    api_client: MedRAGAPIClient, equipment_filter: str, document_filter: Optional[str]
) -> None:
    """
    Render chat interface for querying documents.

    Args:
        api_client: API client instance
        equipment_filter: Equipment filter for queries
        document_filter: Specific document filter (if any)
            - now expected to be full s3 key when selected from S3 browser
    """
    st.markdown("---")
    st.header("💬 Chat with Documents")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        meta = (
                            source.get("metadata", {})
                            if isinstance(source, dict)
                            else {}
                        )
                        equipment = (
                            meta.get("equipment") or source.get("equipment")
                            if isinstance(source, dict)
                            else None
                        )
                        file_name = (
                            meta.get("file_name") or source.get("file_name")
                            if isinstance(source, dict)
                            else None
                        )
                        page_number = (
                            meta.get("page_number") or source.get("page_number")
                            if isinstance(source, dict)
                            else None
                        )
                        content = (
                            source.get("page_content")
                            if isinstance(source, dict)
                            else None
                        )
                        if not content:
                            # Fallback to legacy 'content'
                            content = (
                                source.get("content", "")
                                if isinstance(source, dict)
                                else ""
                            )

                        st.write(f"**Source {i}:**")
                        st.write(f"- Equipment: {equipment or 'Unknown'}")
                        st.write(f"- File: {file_name or 'Unknown'}")
                        st.write(f"- Page: {page_number or 'Unknown'}")
                        st.write(f"- Content: {content[:200]}...")
                        st.write("---")

    # Chat input
    if prompt := st.chat_input("Ask a question about the documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    # Determine equipment filter
                    query_equipment = equipment_filter
                    # document_filter now contains filename (basename)
                    # if a specific file was selected
                    # equipment_filter is already set when the user selects a document,
                    # so prefer it
                    if document_filter and not query_equipment:
                        # As a fallback,
                        # try to extract equipment prefix
                        # if the filter looks like a path
                        parts = document_filter.split("/")
                        if len(parts) > 1:
                            query_equipment = parts[-1]

                    # Make API call — do NOT send file_name or s3_key,
                    # backend expects only equipment
                    response = api_client.query_documents(
                        question=prompt,
                        equipment=document_filter,
                        # query_equipment if query_equipment else None,
                        chat_history=st.session_state.chat_history,
                    )

                    answer = response.get("answer", "Sorry, I couldn't find an answer.")
                    sources = response.get("sources", [])

                    # Further filter sources by specific document if selected
                    # (frontend-side safety net)
                    if document_filter:
                        filtered_sources = []
                        doc_filter_lc = document_filter.lower()
                        for source in sources:
                            if not isinstance(source, dict):
                                continue
                            meta = source.get("metadata", {}) or {}
                            src_file_name = (
                                meta.get("file_name") or source.get("file_name") or ""
                            ).lower()
                            # Prefer metadata 'source' (often holds S3 key) and
                            # 'file_path' as fallbacks
                            src_source_field = (
                                meta.get("source")
                                or meta.get("file_path")
                                or source.get("source")
                                or ""
                            ).lower()
                            # Match by filename or by presence in source/file_path URI
                            # (support full S3 key)
                            if doc_filter_lc and (
                                doc_filter_lc in src_file_name
                                or doc_filter_lc in src_source_field
                            ):
                                filtered_sources.append(source)
                        sources = filtered_sources

                    st.markdown(answer)

                    # Add assistant message to chat history
                    assistant_message = {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                    st.session_state.messages.append(assistant_message)

                    # Update chat history for API
                    st.session_state.chat_history.append(
                        {"role": "user", "content": prompt}
                    )
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )

                    # Show sources
                    if sources:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(sources, 1):
                                meta = (
                                    source.get("metadata", {})
                                    if isinstance(source, dict)
                                    else {}
                                )
                                equipment = (
                                    meta.get("equipment") or source.get("equipment")
                                    if isinstance(source, dict)
                                    else None
                                )
                                file_name = (
                                    meta.get("file_name") or source.get("file_name")
                                    if isinstance(source, dict)
                                    else None
                                )
                                page_number = (
                                    meta.get("page_number") or source.get("page_number")
                                    if isinstance(source, dict)
                                    else None
                                )
                                content = (
                                    source.get("page_content")
                                    if isinstance(source, dict)
                                    else None
                                )
                                if not content:
                                    content = (
                                        source.get("content", "")
                                        if isinstance(source, dict)
                                        else ""
                                    )

                                st.write(f"**Source {i}:**")
                                st.write(f"- Equipment: {equipment or 'Unknown'}")
                                st.write(f"- File: {file_name or 'Unknown'}")
                                st.write(f"- Page: {page_number or 'Unknown'}")
                                st.write(f"- Content: {content[:200]}...")
                                st.write("---")

                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )


def render_clear_chat_button() -> None:
    """Render clear chat history button."""
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.success("Chat history cleared!")
        st.rerun()

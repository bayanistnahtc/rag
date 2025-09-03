import logging
import os
import sys

import streamlit as st

# Ensure project root is on sys.path for reliable imports when
# running in different environments (some runtimes may not honor
# PYTHONPATH for Streamlit subprocess)
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ui.api_client import get_api_client  # noqa: E402
from ui.components import (  # noqa: E402
    render_chat_interface,
    render_clear_chat_button,
    render_document_management,
    render_document_source_selector,
    render_local_document_selector,
    render_s3_document_browser,
)
from ui.s3_client import get_s3_client  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="MedRAG Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📄 MedRAG Assistant")
st.markdown("*Professional Medical Equipment Documentation Assistant*")


def main():
    """Main application logic."""
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API URL configuration
        default_api_base = (
            os.getenv("RAG_API_BASE_URL")
            or os.getenv("RAG_API_URL")
            or "http://medrag-backend:8000"
        )
        api_base_url = st.text_input(
            "RAG API Base URL",
            value=default_api_base,
            help="Base URL for the RAG service API (host:port only)",
        )

        # Initialize API client
        api_client = get_api_client(api_base_url)

        # Document source selection
        document_source = render_document_source_selector()

        # Document selection based on source type
        equipment_filter = ""
        document_filter = None

        if document_source == "local":
            selected_source = render_local_document_selector()
            if selected_source:
                st.info(f"📄 Selected: **{os.path.basename(selected_source)}**")
        else:
            # S3 document browser
            s3_client = get_s3_client()
            equipment_filter, document_filter = render_s3_document_browser(s3_client)

            # Document management for S3
            render_document_management(api_client)

        # Clear chat button
        render_clear_chat_button()

        st.markdown("---")
        st.header("📊 System Status")

        # Show current index size
        try:
            index_size = api_client.get_index_size()
            if index_size is not None:
                st.success(f"📚 Index: {index_size} chunks")
            else:
                st.warning("📚 Index: Status unknown")
        except Exception:
            st.error("📚 Index: Connection failed")

        st.info("💡 This app provides access to medical equipment documentation.")

    # Main chat interface
    render_chat_interface(api_client, equipment_filter, document_filter)


if __name__ == "__main__":
    main()

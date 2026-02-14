# Entry point for streamlit run app.py

import streamlit as st
from app_core.state import init_session_state
from app_core.ui import render_query_page, render_sidebar, render_upload_page

st.set_page_config(
    page_title="Semantic Search Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    # Initialize state vars referenced throughout the app session
    init_session_state()
    
    # Create left sidebar used for page nav (determines main page render)
    # Also contains RAG configs
    settings = render_sidebar()

    # Render page based on whether Upload Docs or Query Engine page selected
    if settings["page"] == "Upload Documents":
        render_upload_page()
    else:
        render_query_page(settings)


if __name__ == "__main__":
    main()

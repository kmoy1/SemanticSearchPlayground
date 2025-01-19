import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from model import load_embeddings_model, create_faiss_store

# st.title("Semantic Search Playground")
# st.write("Upload your documents and query them semantically.")
st.set_page_config(page_title="Semantic Search Engine", layout="wide", initial_sidebar_state="expanded")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload Documents", "Query Engine"])

# Document Storage Initialization.
if "documents" not in st.session_state:
    st.session_state.documents = {}


if page == "Upload Documents":
    st.title("📄 Upload Documents")
    st.write("Upload text documents to the model. These are the training data of the semantic search engine.")
    uploaded_files = st.file_uploader("", type=["txt"], accept_multiple_files=True)

    # File uploading logic.
    if uploaded_files:
        with st.spinner("Processing uploaded documents..."):
            for file in uploaded_files:
                st.session_state.documents[file.name] = file.read().decode("utf-8")
        st.success("Documents uploaded successfully!")
    if st.session_state.documents:
        st.subheader("Uploaded Documents:")
        for doc_name, doc_content in st.session_state.documents.items():
            with st.expander(doc_name):
                st.write(doc_content)

elif page == "Query Engine":
    st.title("🔍 Query Semantic Search Engine")
    if not st.session_state.documents:
        st.warning("No documents uploaded yet! Please upload documents in the 'Upload Documents' section.")
    else:  
        # Display uploaded files (1+)
        st.subheader("Uploaded Files")
        st.write("These documents serve as the training data for your model.")
        for doc_name, doc_content in st.session_state.documents.items():
            with st.expander(doc_name):  # Expandable sections for each document
                st.write(doc_content)
        
        # Check if model has already been trained.
        if "faiss_store" not in st.session_state:
            if st.button("Train Model"):
                with st.spinner("Creating FAISS store (training the model)..."):
                    embeddings_model = load_embeddings_model()
                    documents = [
                        Document(page_content=doc_content, metadata={"file_name": doc_name})
                        for doc_name, doc_content in st.session_state.documents.items()
                    ]
                    st.session_state.faiss_store = create_faiss_store(documents, embeddings_model)
                st.success("Model Trained Successfully.")
        # Only display query search box after model is trained.
        if "faiss_store" in st.session_state:
            query = st.text_input("Query Model:")
            if query:
                with st.spinner("Querying model for search query " + query):
                    results = st.session_state.faiss_store.similarity_search(query, k=1)
                st.subheader("Model Found Most Similar Documents:")
                for result in results:
                    st.markdown(f"**📄 {result.metadata['file_name']}**")
                    st.write(result.page_content)
                    st.write("---")
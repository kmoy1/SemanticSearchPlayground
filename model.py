from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS


def load_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")

def create_faiss_store(documents, embeddings_model):
    """Create FAISS store."""
    return FAISS.from_documents(documents, embeddings_model)

def create_documents(uploaded_files):
    """Create list of langchain Document objects from uploaded files."""
    documents = []
    for file in uploaded_files:
        file_content = file.read().decode("utf-8")  # Read and decode the file
        documents.append(Document(page_content=file_content, metadata={"file_name": file.name}))
    return documents
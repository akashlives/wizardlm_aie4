from langchain_community.document_loaders import PyMuPDFLoader
from typing import Any
from .state_config import QAState


def load_documents_and_generate_embeddings(state: QAState) -> QAState:
    """
    Load documents from a PDF file and generate embeddings for the document content.

    This function loads documents from a specified PDF file, generates embeddings
    for the document content, and updates the state with the loaded documents
    and their embeddings.

    Args:
        state (QAState): The current state of the QA system, containing pdf_path and embedding_model.

    Returns:
        QAState: The updated state with the loaded documents and their embeddings.

    Raises:
        ValueError: If no documents were loaded from the PDF or if pdf_path or embedding_model is missing.
    """
    pdf_path = state.get("pdf_path")
    embedding_model = state.get("embedding_model")

    if not pdf_path or not embedding_model:
        raise ValueError("pdf_path and embedding_model must be provided in the state.")

    # Load documents
    loader = PyMuPDFLoader(file_path=pdf_path)
    documents = loader.load()

    if not documents:
        raise ValueError("No documents were loaded from the PDF.")

    # Generate embeddings
    document_texts = [doc.page_content for doc in documents]
    document_embeddings = embedding_model.embed_documents(document_texts)

    # Update state
    state["documents"] = documents
    state["document_embeddings"] = document_embeddings

    return state

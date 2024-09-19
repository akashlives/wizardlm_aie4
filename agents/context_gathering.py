import faiss
import numpy as np
from typing import Any, List, Dict
from .state_config import QAState
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue


def context_gathering(
    state: QAState,
    k: int = 5,
    use_qdrant: bool = False,
    qdrant_client: QdrantClient = None,
    collection_name: str = None,
) -> QAState:
    """
    Handles context dynamically using FAISS or Qdrant for evolved questions.

    Args:
        state (QAState): The current state of the QA system.
        k (int): The number of relevant contexts to retrieve for each question.
        use_qdrant (bool): Whether to use Qdrant instead of FAISS.
        qdrant_client (QdrantClient): Qdrant client instance (required if use_qdrant is True).
        collection_name (str): Name of the Qdrant collection (required if use_qdrant is True).

    Returns:
        QAState: The updated state with relevant contexts for each evolved question.
    """
    embedding_model = state.get("embedding_model")
    document_embeddings = state.get("document_embeddings", [])
    documents = state.get("documents", [])
    evolved_questions = state.get("evolved_questions", [])

    if not document_embeddings and not use_qdrant:
        raise ValueError("Document embeddings are missing from the state.")

    if use_qdrant and (qdrant_client is None or collection_name is None):
        raise ValueError(
            "Qdrant client and collection name are required when using Qdrant."
        )

    contexts = []

    if use_qdrant:

        def search_func(query_vector):
            return qdrant_search(qdrant_client, collection_name, query_vector, k)

    else:
        index = faiss.IndexFlatL2(len(document_embeddings[0]))
        index.add(np.array(document_embeddings).astype("float32"))

        def search_func(query_vector):
            return faiss_search(index, query_vector, k)

    for q in evolved_questions:
        query_vector = embedding_model.embed_query(q["evolved_question"])
        relevant_indices = search_func(query_vector)

        relevant_contexts = [
            documents[i].page_content for i in relevant_indices if i < len(documents)
        ]

        if relevant_contexts:
            contexts.append(
                {
                    "id": q["id"],
                    "question": q["evolved_question"],
                    "contexts": relevant_contexts,
                }
            )
        else:
            print(f"No contexts found for question ID {q['id']}.")

    state["contexts"] = contexts
    return state


def faiss_search(index: faiss.Index, query_vector: np.ndarray, k: int) -> List[int]:
    """
    Perform a k-nearest neighbors search using FAISS.

    Args:
        index (faiss.Index): The FAISS index to search.
        query_vector (np.ndarray): The query vector to search for.
        k (int): The number of nearest neighbors to retrieve.

    Returns:
        List[int]: The indices of the k-nearest neighbors.
    """
    _, indices = index.search(np.array([query_vector]).astype("float32"), k=k)
    return indices[0].tolist()


def qdrant_search(
    client: QdrantClient, collection_name: str, query_vector: List[float], k: int
) -> List[int]:
    """
    Perform a k-nearest neighbors search using Qdrant.

    Args:
        client (QdrantClient): The Qdrant client instance.
        collection_name (str): The name of the Qdrant collection.
        query_vector (List[float]): The query vector to search for.
        k (int): The number of nearest neighbors to retrieve.

    Returns:
        List[int]: The indices of the k-nearest neighbors.
    """
    search_result = client.search(
        collection_name=collection_name, query_vector=query_vector, limit=k
    )
    return [hit.id for hit in search_result]

from typing import TypedDict, Optional, List, Any


class QAState(TypedDict):
    pdf_path: Optional[str]
    embedding_model: Optional[Any]
    model: Optional[Any]
    critic_model: Optional[Any]
    documents: Optional[List]
    document_embeddings: Optional[List]
    questions: Optional[List[dict]]
    evolved_questions: Optional[List[dict]]
    answers: Optional[List[dict]]
    contexts: Optional[List[dict]]
    final_output: Optional[List[dict]]
    max_evolved_questions: Optional[int]
    max_evolutions_per_technique: Optional[int]

from typing import List, Dict, Optional, Any
from .state_config import QAState


def export_agent(state: QAState) -> QAState:
    """
    Exports the final output including questions, evolutions, answers, and contexts.

    This function consolidates initial questions, evolved questions, their answers, and relevant contexts into
    a structured format for easy parsing and analysis.

    Args:
        state (QAState): The current state of the QA system containing questions, evolved questions,
                         answers, and contexts.

    Returns:
        QAState: The updated state with the final_output field added, containing the consolidated data.

    Raises:
        KeyError: If required fields are missing from the state.
    """
    initial_questions: List[Dict[str, Any]] = state.get("questions", [])
    evolved_questions: List[Dict[str, Any]] = state.get("evolved_questions", [])
    answers: List[Dict[str, Any]] = state.get("answers", [])
    contexts: List[Dict[str, Any]] = state.get("contexts", [])

    final_output: List[Dict[str, Any]] = []

    def find_answer_and_context(question_id: str) -> Dict[str, Optional[Any]]:
        """
        Finds the answer and context for a given question ID.

        Args:
            question_id (str): The ID of the question to find the answer and context for.

        Returns:
            Dict[str, Optional[Any]]: A dictionary containing the answer and context, if found.
        """
        answer = next((a for a in answers if a["id"] == question_id), None)
        context = next((c for c in contexts if c["id"] == question_id), None)
        return {
            "answer": answer["answer"] if answer else None,
            "context": context["contexts"] if context else None,
        }

    for eq in evolved_questions:
        answer_context = find_answer_and_context(eq["id"])
        final_output.append(
            {
                "id": eq["id"],
                "evolution_type": eq["evolution_type"],
                "answer": answer_context["answer"],
                "contexts": answer_context["context"],
                "evolved_question": eq["evolved_question"],
            }
        )

    # Debug: Print final output summary
    print(f"Final Output Generated: {len(final_output)} entries")
    for entry in final_output:
        print(
            f"ID: {entry['id']}, Type: {entry['evolution_type']}, "
            f"Answer: {'Present' if entry['answer'] else 'Missing'}, "
            f"Contexts: {'Present' if entry['contexts'] else 'Missing'}"
        )

    state["final_output"] = final_output
    return state

from langchain_core.prompts import PromptTemplate
from .state_config import QAState
from typing import List, Dict, Tuple
import uuid
import random
from .evolution_techniques import evolution_techniques


def apply_evolution(
    question: str,
    context: str,
    instruction: str,
    examples: List[Dict],
    prompt_template: PromptTemplate,
    model,
) -> str:
    """
    Applies an evolution technique to generate a question using a language model.

    Args:
        question (str): The original question or a prompt to generate a question.
        context (str): The context for the question.
        instruction (str): The instruction for evolving the question.
        examples (List[Dict]): A list of example evolutions.
        prompt_template (PromptTemplate): The prompt template for evolution.
        model: The language model to use.

    Returns:
        str: The evolved question.
    """
    input_dict = {"context": context, "instruction": instruction, "examples": examples}
    if "question" in prompt_template.input_variables:
        input_dict["question"] = (
            question if question else "Generate a question about the following context:"
        )
    prompt = prompt_template.format(**input_dict)
    result = model.invoke(prompt)
    return result.content.strip() if hasattr(result, "content") else str(result).strip()


def create_evolved_question_dict(
    question_id: str, evolution_type: str, evolved_question: str
) -> Dict[str, str]:
    """
    Creates a dictionary representing an evolved question.

    Args:
        question_id (str): The ID of the original question.
        evolution_type (str): The type of evolution applied.
        evolved_question (str): The evolved question.

    Returns:
        Dict[str, str]: A dictionary containing the evolved question information.
    """
    return {
        "id": f"{question_id}_{evolution_type}_{uuid.uuid4().hex[:8]}",
        "original_question_id": question_id,
        "evolved_question": evolved_question,
        "evolution_type": evolution_type,
    }


def generate_evolved_questions(
    documents: List[Dict],
    evolution_techniques: List[Tuple[str, PromptTemplate, float]],
    model,
    max_evolved_questions: int = 10,
    max_evolutions_per_technique: int = 5,
) -> List[Dict]:
    evolved_questions = []

    for technique in evolution_techniques:
        name, prompt_template, _ = technique
        evolutions = 0
        while evolutions < max_evolutions_per_technique:
            document = random.choice(documents)
            context = document.page_content
            name, prompt_template, _ = technique

            evolved_question = apply_evolution(
                "",  # Empty string for initial question generation
                context,
                prompt_template.template,
                prompt_template.input_variables,
                prompt_template,
                model,
            )

            if evolved_question:
                evolved_questions.append(
                    create_evolved_question_dict(
                        str(uuid.uuid4()), name, evolved_question
                    )
                )
                evolutions += 1

            if len(evolved_questions) >= max_evolved_questions:
                return evolved_questions

    return evolved_questions


def evolution_agent(
    state: QAState,
    model,
    evolution_techniques: List[Tuple[str, PromptTemplate, float]],
    max_evolved_questions: int = 10,
    max_evolutions_per_question: int = 5,
) -> QAState:
    """
    Generates evolved questions using various techniques without initial questions.

    Args:
        state (QAState): The current state of the QA system.
        model: The language model to use for evolution.
        evolution_techniques (List[Tuple[str, PromptTemplate, float]]): List of evolution techniques.
        max_evolved_questions (int): Maximum number of total evolved questions to generate.
        max_evolutions_per_question (int): Maximum number of evolutions to generate per technique.

    Returns:
        QAState: The updated state with evolved questions.
    """
    documents = state.get("documents", [])
    if not documents:
        raise ValueError("No documents found in the state to generate questions from.")

    evolved_questions = generate_evolved_questions(
        documents,
        evolution_techniques,
        model,
        max_evolved_questions,
        max_evolutions_per_question,
    )

    state["evolved_questions"] = evolved_questions
    return state

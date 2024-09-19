from typing import Dict
from .state_config import QAState
from .evolution_agent import evolution_agent
from .question_critic_agent import critic_agent
from .evolution_techniques import evolution_techniques
import random
import uuid
from .evolution_agent import apply_evolution


def question_generation_pipeline(
    state: QAState,
    evolution_distribution: Dict[str, float] = None,
    max_evolved_questions: int = 12,
    max_evolutions_per_technique: int = 5,
    quality_threshold: int = 3,
) -> QAState:
    """
    Generates and validates evolved questions based on the input state.

    This pipeline applies evolution techniques to generate new questions and critiques them.

    Args:
        state (QAState): The current state of the QA system.
        evolution_distribution (Dict[str, float]): A dictionary of evolution techniques with their distributions.
        max_evolved_questions (int): Maximum number of evolved questions to generate.
        max_evolutions_per_technique (int): Maximum number of evolutions per technique.
        quality_threshold (int): Minimum quality score for a question to be considered valid.

    Returns:
        QAState: The updated state with evolved and validated questions.
    """
    max_evolved_questions = state.get("max_evolved_questions") or max_evolved_questions
    max_evolutions_per_technique = (
        state.get("max_evolutions_per_technique") or max_evolutions_per_technique
    )

    if evolution_distribution is None:
        evolution_distribution = {
            technique[0]: 1 / len(evolution_techniques)
            for technique in evolution_techniques
        }

    # Get models from state
    model = state.get("model")
    critic_model = state.get("critic_model")

    if not model or not critic_model:
        raise ValueError("Model or critic model not found in the state.")

    # Use the imported evolution_techniques directly
    evolution_techniques_with_distribution = [
        (name, prompt_template, evolution_distribution.get(name, 0))
        for name, prompt_template in evolution_techniques
    ]

    # Run evolution agent
    state = evolution_agent(
        state,
        model,
        evolution_techniques_with_distribution,
        max_evolved_questions,
        max_evolutions_per_technique,
    )

    # Run critic agent
    state = critic_agent(
        state,
        critic_model,
        threshold=quality_threshold,
        max_validated_questions=max_evolved_questions,
    )

    return state


def generate_initial_questions(state: QAState, model, num_questions: int) -> QAState:
    """
    Generates initial questions from documents using the provided model.
    """
    documents = state.get("documents", [])
    if not documents:
        raise ValueError("No documents found in the state to generate questions from.")

    simple_question_technique = next(
        (
            technique
            for technique in evolution_techniques
            if technique[0] == "simple_question"
        ),
        None,
    )
    if not simple_question_technique:
        raise ValueError("Simple question evolution technique not found.")

    initial_questions = []
    for _ in range(num_questions):
        document = random.choice(documents)
        context = document.page_content
        if context:
            name, prompt_template = simple_question_technique
            question = apply_evolution(
                "",
                context,
                prompt_template.template,
                prompt_template.input_variables,
                prompt_template,
                model,
            )
            if question:
                initial_questions.append(
                    {
                        "id": str(uuid.uuid4()),
                        "question": question,
                        "context": context,
                    }
                )

    if not initial_questions:
        raise ValueError("Failed to generate any initial questions.")

    state["questions"] = initial_questions
    return state

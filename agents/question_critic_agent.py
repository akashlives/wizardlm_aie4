from .state_config import QAState
from typing import List, Dict
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser
import json


def create_critic_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["question"],
        template="""
        Critique the synthetically generated question based on the following rubrics. Provide a score for each rubric: Independence and Clear Intent. Scores are given as low (0), medium (1), or high (2).
        You must always output a JSON object with 'Independence' and 'Clear Intent' keys."

        Examples:
        1. Question: "How does AI improve efficiency and accuracy across different industries?"
           Feedback: {{"Independence": 2, "Clear Intent": 2}}

        2. Question: "Explain the benefits of AI."
           Feedback: {{"Independence": 1, "Clear Intent": 1}}

        3. Question: "How does AI?"
           Feedback: {{"Independence": 0, "Clear Intent": 0}}

        Question: {question}

        Feedback:
        """,
    )


def validate_question(question: Dict, prompt_template: PromptTemplate, model) -> Dict:
    print("Validating question: ", question)
    prompt = prompt_template.format(question=question["evolved_question"])
    chain = model | SimpleJsonOutputParser()
    result = chain.invoke(prompt)
    print(f"Result: {result}")

    try:
        feedback = result
        if "Independence" not in feedback or "Clear Intent" not in feedback:
            raise ValueError("Feedback does not contain required keys")
        return feedback
    except (json.JSONDecodeError, ValueError) as e:
        print(
            f"Error parsing feedback for question ID {question.get('id', 'unknown')}: {e}"
        )
        return {"Independence": 0, "Clear Intent": 0}


def critic_agent(
    state: QAState, model, threshold: int = 3, max_validated_questions: int = 10
) -> QAState:
    """
    Uses a critic model to validate the evolved questions.

    Args:
        state (QAState): The current state of the QA system.
        model: The language model to use for criticism.
        threshold (int): Minimum total score required for a question to be considered valid.
        max_validated_questions (int): Maximum number of validated questions to keep.

    Returns:
        QAState: The updated state with validated questions.
    """
    evolved_questions = state.get("evolved_questions", [])
    critic_prompt = create_critic_prompt()
    validated_questions = []

    for q in evolved_questions:
        feedback = validate_question(q, critic_prompt, model)
        total_score = feedback["Independence"] + feedback["Clear Intent"]

        if total_score >= threshold:
            q["critic_feedback"] = feedback
            validated_questions.append(q)

        if len(validated_questions) >= max_validated_questions:
            break

    state["validated_questions"] = validated_questions
    return state

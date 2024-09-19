from typing import Any, List
from .state_config import QAState
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import Runnable


def create_answer_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["question", "context"],
        template="""
        Provide a detailed answer for the following question using the given context:
        Answer the question in paragraph format, don't use font styles or bullet points.

        Question: {question}
        Context: {context}

        Answer:
        """,
    )


def generate_answer(
    question: str, context: str, prompt_template: PromptTemplate, model: Any
) -> str:
    input_dict = {"question": question, "context": context}
    prompt = prompt_template.format(**input_dict)
    result = model.invoke(prompt)
    return result.content.strip() if hasattr(result, "content") else str(result).strip()


def answer_generator(
    state: QAState,
    max_answers: int = 10,
) -> QAState:
    model = state.get("model")
    evolved_questions = state.get("evolved_questions", [])
    contexts = state.get("contexts", [])
    answers = []
    answer_prompt = create_answer_prompt()

    for q in evolved_questions:
        if len(answers) >= max_answers:
            break

        context_data = next(
            (c["contexts"] for c in contexts if c["id"] == q["id"]), None
        )
        if not context_data:
            print(
                f"No context found for question ID {q['id']}. Marking as research required."
            )
            answers.append(
                {
                    "id": q["id"],
                    "question": q["evolved_question"],
                    "answer": "Research required: Insufficient context to provide an accurate answer.",
                    "context": "",
                }
            )
            continue

        combined_context = " ".join(context_data)
        answer = generate_answer(
            q["evolved_question"], combined_context, answer_prompt, model
        )

        if answer:
            answers.append(
                {
                    "id": q["id"],
                    "question": q["evolved_question"],
                    "answer": answer,
                    "context": combined_context,
                }
            )
        else:
            print(
                f"No answer generated for question ID {q['id']}. Marking as research required."
            )
            answers.append(
                {
                    "id": q["id"],
                    "question": q["evolved_question"],
                    "answer": "Research required: Unable to generate an answer based on the given context.",
                    "context": combined_context,
                }
            )

    state["answers"] = answers
    return state

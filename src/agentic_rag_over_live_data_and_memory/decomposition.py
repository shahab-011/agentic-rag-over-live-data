
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()


class DecomposedQuestion(BaseModel):
    original_question: str = Field(
        description="The user's original question, unchanged."
    )
    subquestions: list[str] = Field(
        min_length=1,
        description="Focused, non-redundant questions needed to answer the original question."
    )


SYSTEM_PROMPT = """
You are the question decomposition component of an
Agentic Work Intelligence Assistant.

Break the user's request into focused subquestions that
can later be routed to GitHub, Gmail, Notion, Jira, or a combination.

Rules:
1. Preserve the user's original intent and constraints.
2. Split complex requests into focused subquestions.
3. Avoid duplicate or unnecessary questions.
4. Preserve relevant project names and time ranges.
5. Do not answer the questions.
6. Do not retrieve information or call tools.
7. Never invent facts the user has not provided.
8. If the question is simple, return it as one subquestion.
9. Do not turn relative words such as "recently" or "latest" into
   clarification questions. Preserve them in the subquestion so the source
   can return the newest matching records without inventing a time range.

Return data matching the requested output schema.
"""


def decompose_question(question: str) -> dict:
    question = question.strip()

    if not question:
        raise ValueError("The question cannot be empty.")

    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from .env.")

    if not model:
        raise ValueError("GROQ_MODEL is missing from .env.")

    llm = ChatGroq(
        model=model,
        api_key=api_key,
        temperature=0,
    )

    structured_llm = llm.with_structured_output(
        DecomposedQuestion
    )

    result = structured_llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", question),
        ]
    )

    return result.model_dump()


if __name__ == "__main__":
    user_question = input("Enter your question: ")

    try:
        result = decompose_question(user_question)

        print("\nDecomposed questions:\n")
        print(f"Original: {result['original_question']}")

        for index, subquestion in enumerate(
            result["subquestions"], start=1
        ):
            print(f"{index}. {subquestion}")

    except Exception as exc:
        print(f"\nDecomposition failed: {exc}")

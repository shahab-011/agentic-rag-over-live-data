
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()


Source = Literal["github", "gmail", "notion", "jira"]


class RouteDecision(BaseModel):
    subquestion: str = Field(
        description="The subquestion being routed."
    )
    sources: list[Source] = Field(
        min_length=1,
        description=(
            "Data sources required to answer the subquestion. "
            "Use github, gmail, notion, jira, or any combination."
        ),
    )


class RoutingResult(BaseModel):
    routes: list[RouteDecision] = Field(
        min_length=1,
        description="A routing decision for every input subquestion.",
    )


SYSTEM_PROMPT = """
You are the source-routing component of an Agentic RAG system.

The system can retrieve information from four sources:
- github: repository activity, commits, code changes, issues,
  pull requests, reviews, project tasks, and the authenticated
  user's repository profile recorded in GitHub.
- gmail: email discussions, decisions, approvals, commitments,
  deadlines, and blockers mentioned in emails.
- notion: project documentation, notes, plans, and decisions in
  pages shared with the Notion integration.
- jira: Jira projects, issues, statuses, assignees, and issue details
  visible to the authenticated Jira account.

For each subquestion, choose the source or sources needed to
answer it.

Rules:
1. Use github for questions about repository activity, issues,
   pull requests, code, GitHub project tasks, and the user's
   own GitHub repositories or profile.
2. Use gmail for questions about email conversations, decisions,
   approvals, commitments, and email-mentioned deadlines.
3. Use multiple sources when answering requires connecting information
   from more than one source, including Notion project documentation.
   Use jira for questions about Jira issues or projects.
4. Choose sources based on the information needed, not merely
   keywords in the question.
5. Do not answer the subquestions.
6. Do not invent facts or call external tools.
7. Include exactly one routing decision for each input subquestion.
8. Do not add subquestions that were not provided.

Return data matching the requested output schema.
"""


def route_subquestions(subquestions: list[str]) -> dict:
    cleaned_questions = [
        question.strip()
        for question in subquestions
        if question and question.strip()
    ]

    if not cleaned_questions:
        raise ValueError("At least one subquestion is required.")

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

    structured_llm = llm.with_structured_output(RoutingResult)

    result = structured_llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Route these subquestions:\n"
                + "\n".join(
                    f"{index}. {question}"
                    for index, question in enumerate(
                        cleaned_questions, start=1
                    )
                ),
            ),
        ]
    )

    # Ensure every input question has exactly one decision.
    returned_questions = [
        decision.subquestion for decision in result.routes
    ]

    if len(returned_questions) != len(cleaned_questions):
        raise ValueError(
            "The router did not return one decision per subquestion."
        )

    return result.model_dump()


if __name__ == "__main__":
    sample_questions = [
        "What recent code changes were made in the repository?",
        "What decisions were discussed in project emails?",
        (
            "Which open GitHub issues are blocked by decisions "
            "mentioned in emails?"
        ),
    ]

    try:
        result = route_subquestions(sample_questions)

        print("\nSource routing results:\n")

        for decision in result["routes"]:
            sources = " + ".join(decision["sources"])
            print(f"Question: {decision['subquestion']}")
            print(f"Sources:  {sources}\n")

    except Exception as exc:
        print(f"\nRouting failed: {exc}")

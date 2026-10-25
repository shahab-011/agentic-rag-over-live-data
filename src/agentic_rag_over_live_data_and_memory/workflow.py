
import asyncio
import json
import operator
import sys
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agentic_rag_over_live_data_and_memory.answer_generation import (
    generate_cited_answer,
)
from agentic_rag_over_live_data_and_memory.decomposition import (
    decompose_question,
)
from agentic_rag_over_live_data_and_memory.github_retriever import (
    retrieve_github,
)
from agentic_rag_over_live_data_and_memory.jira_retriever import (
    retrieve_jira,
)
from agentic_rag_over_live_data_and_memory.notion_retriever import (
    retrieve_notion,
)
from agentic_rag_over_live_data_and_memory.router import (
    route_subquestions,
)
from agentic_rag_over_live_data_and_memory.schemas import (
    Citation,
    RetrievalResult,
    WorkflowOutput,
)


class WorkflowState(TypedDict, total=False):
    original_question: str
    subquestions: list[str]
    routes: list[dict[str, Any]]
    retrieval_tasks: list[dict[str, str]]
    task_index: int
    current_task: dict[str, str] | None
    retrieval_results: Annotated[list[RetrievalResult], operator.add]
    final_answer: str | None
    citations: list[Citation]


def build_workflow_graph():
    """Build the LangGraph that routes questions to their source retrievers."""

    def decompose_node(state: WorkflowState) -> dict[str, Any]:
        decomposition = decompose_question(state["original_question"])
        return {"subquestions": decomposition["subquestions"]}

    def route_node(state: WorkflowState) -> dict[str, Any]:
        subquestions = state["subquestions"]
        routing = route_subquestions(subquestions)
        routes = routing["routes"]
        retrieval_tasks = [
            {"subquestion": decision["subquestion"], "source": source}
            for decision in routes
            for source in decision["sources"]
        ]
        return {
            "routes": routes,
            "retrieval_tasks": retrieval_tasks,
            "task_index": 0,
        }

    def prepare_next_task_node(state: WorkflowState) -> dict[str, Any]:
        tasks = state["retrieval_tasks"]
        task_index = state["task_index"]
        if task_index >= len(tasks):
            return {"current_task": None}
        return {
            "current_task": tasks[task_index],
            "task_index": task_index + 1,
        }

    def select_source(state: WorkflowState) -> str:
        task = state.get("current_task")
        if task is None:
            return "done"
        source = task["source"]
        if source not in {"github", "gmail", "notion", "jira"}:
            raise ValueError(f"Unsupported retrieval source: {source}")
        return source

    async def retrieve_github_node(
        state: WorkflowState,
    ) -> dict[str, list[RetrievalResult]]:
        task = state["current_task"]
        if task is None:
            raise RuntimeError("GitHub retrieval was selected without a task.")
        subquestion = task["subquestion"]
        try:
            evidence = await retrieve_github(subquestion)
            result = RetrievalResult(
                subquestion=subquestion,
                source="github",
                status="success",
                result=evidence,
            )
        except Exception as exc:  # noqa: BLE001 - report source failures in the workflow result.
            result = RetrievalResult(
                subquestion=subquestion,
                source="github",
                status="error",
                error=str(exc),
            )
        return {"retrieval_results": [result]}

    def retrieve_gmail_node(
        state: WorkflowState,
    ) -> dict[str, list[RetrievalResult]]:
        task = state["current_task"]
        if task is None:
            raise RuntimeError("Gmail retrieval was selected without a task.")
        result = RetrievalResult(
            subquestion=task["subquestion"],
            source="gmail",
            status="not_implemented",
            error="Gmail retrieval has not been connected yet.",
        )
        return {"retrieval_results": [result]}

    async def retrieve_notion_node(
        state: WorkflowState,
    ) -> dict[str, list[RetrievalResult]]:
        task = state["current_task"]
        if task is None:
            raise RuntimeError("Notion retrieval was selected without a task.")
        subquestion = task["subquestion"]
        try:
            evidence = await retrieve_notion(subquestion)
            result = RetrievalResult(
                subquestion=subquestion,
                source="notion",
                status="success",
                result=evidence,
            )
        except Exception as exc:  # noqa: BLE001 - report source failures in the workflow result.
            result = RetrievalResult(
                subquestion=subquestion,
                source="notion",
                status="error",
                error=str(exc),
            )
        return {"retrieval_results": [result]}

    async def retrieve_jira_node(
        state: WorkflowState,
    ) -> dict[str, list[RetrievalResult]]:
        task = state["current_task"]
        if task is None:
            raise RuntimeError("Jira retrieval was selected without a task.")
        subquestion = task["subquestion"]
        try:
            evidence = await retrieve_jira(subquestion)
            result = RetrievalResult(
                subquestion=subquestion,
                source="jira",
                status="success",
                result=evidence,
            )
        except Exception as exc:  # noqa: BLE001 - report source failures in the workflow result.
            result = RetrievalResult(
                subquestion=subquestion,
                source="jira",
                status="error",
                error=str(exc),
            )
        return {"retrieval_results": [result]}

    def generate_answer_node(state: WorkflowState) -> dict[str, Any]:
        final_answer, citations = generate_cited_answer(
            state["original_question"],
            state.get("retrieval_results", []),
        )
        return {"final_answer": final_answer, "citations": citations}

    graph = StateGraph(WorkflowState)
    graph.add_node("decompose", decompose_node)
    graph.add_node("route", route_node)
    graph.add_node("prepare_next_task", prepare_next_task_node)
    graph.add_node("retrieve_github", retrieve_github_node)
    graph.add_node("retrieve_gmail", retrieve_gmail_node)
    graph.add_node("retrieve_notion", retrieve_notion_node)
    graph.add_node("retrieve_jira", retrieve_jira_node)
    graph.add_node("generate_answer", generate_answer_node)

    graph.add_edge(START, "decompose")
    graph.add_edge("decompose", "route")
    graph.add_edge("route", "prepare_next_task")
    graph.add_conditional_edges(
        "prepare_next_task",
        select_source,
        {
            "github": "retrieve_github",
            "gmail": "retrieve_gmail",
            "notion": "retrieve_notion",
            "jira": "retrieve_jira",
            "done": "generate_answer",
        },
    )
    graph.add_edge("retrieve_github", "prepare_next_task")
    graph.add_edge("retrieve_gmail", "prepare_next_task")
    graph.add_edge("retrieve_notion", "prepare_next_task")
    graph.add_edge("retrieve_jira", "prepare_next_task")
    graph.add_edge("generate_answer", END)
    return graph.compile()


workflow_graph = build_workflow_graph()


async def run_workflow(question: str) -> WorkflowOutput:
    """Run the LangGraph workflow for a user question."""

    question = question.strip()

    if not question:
        raise ValueError("Please enter a question.")

    result = await workflow_graph.ainvoke(
        {
            "original_question": question,
            "retrieval_results": [],
        }
    )
    return WorkflowOutput(
        original_question=result["original_question"],
        subquestions=result["subquestions"],
        routes=result["routes"],
        retrieval_results=result.get("retrieval_results", []),
        final_answer=result.get("final_answer"),
        citations=result.get("citations", []),
    )


def print_workflow_result(result: WorkflowOutput) -> None:
    """Print the workflow response in a readable format."""

    print("\n========== WORKFLOW RESULT ==========\n")

    print("Original question:")
    print(result.original_question)

    print("\n--- Decomposed Questions ---")

    for index, subquestion in enumerate(
        result.subquestions, start=1
    ):
        print(f"{index}. {subquestion}")

    print("\n--- Source Routing ---")

    for decision in result.routes:
        print(f"\nQuestion: {decision['subquestion']}")
        print(f"Sources: {', '.join(decision['sources'])}")

    print("\n--- Retrieval Results ---")

    for item in result.retrieval_results:
        print(f"\nQuestion: {item.subquestion}")
        print(f"Source: {item.source}")
        print(f"Status: {item.status}")

        if item.result is not None:
            print(
                json.dumps(
                    item.result,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            )

        if item.error:
            print(f"Error: {item.error}")

    print("\n--- Final Answer ---")
    print(result.final_answer or "No answer was generated.")

    if result.citations:
        print("\n--- Citations ---")
        for citation in result.citations:
            print(
                f"[{citation.number}] {citation.source} — "
                f"{citation.subquestion}: {citation.url}"
            )

    print("\n--- Full Structured Result ---")
    print(result.model_dump_json(indent=2))


async def main() -> None:
    """Run the interactive project workflow."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")

    user_question = input("Ask your project assistant: ")

    try:
        result = await run_workflow(user_question)
        print_workflow_result(result)

    except ValueError as exc:
        print(f"\nInvalid input: {exc}")

    except Exception as exc:  # noqa: BLE001 - show unexpected errors to the CLI user.
        print(f"\nWorkflow failed: {exc}")


if __name__ == "__main__":
    asyncio.run(main())

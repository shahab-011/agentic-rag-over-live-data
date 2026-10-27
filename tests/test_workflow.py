import asyncio

from agentic_rag_over_live_data_and_memory import workflow


def test_langgraph_routes_each_source_and_collects_results(monkeypatch) -> None:
    github_questions = []
    jira_questions = []
    notion_questions = []
    gmail_questions = []
    answer_input = {}

    monkeypatch.setattr(
        workflow,
        "decompose_question",
        lambda question: {
            "original_question": question,
            "subquestions": ["GitHub question", "cross-source question"],
        },
    )
    monkeypatch.setattr(
        workflow,
        "route_subquestions",
        lambda questions: {
            "routes": [
                {"subquestion": questions[0], "sources": ["github"]},
                {
                    "subquestion": questions[1],
                    "sources": ["github", "gmail", "notion", "jira"],
                },
            ]
        },
    )

    async def fake_retrieve_github(subquestion: str) -> dict:
        github_questions.append(subquestion)
        return {"evidence": {"question": subquestion}}

    async def fake_retrieve_notion(subquestion: str) -> dict:
        notion_questions.append(subquestion)
        return {"evidence": {"question": subquestion}}

    async def fake_retrieve_jira(subquestion: str) -> dict:
        jira_questions.append(subquestion)
        return {"evidence": {"question": subquestion}}

    async def fake_retrieve_gmail(subquestion: str) -> dict:
        gmail_questions.append(subquestion)
        return {"evidence": {"question": subquestion}}

    def fake_generate_answer(question, retrieval_results):
        answer_input["question"] = question
        answer_input["retrieval_results"] = retrieval_results
        return "Grounded answer.", []

    monkeypatch.setattr(workflow, "retrieve_github", fake_retrieve_github)
    monkeypatch.setattr(workflow, "retrieve_gmail", fake_retrieve_gmail)
    monkeypatch.setattr(workflow, "retrieve_notion", fake_retrieve_notion)
    monkeypatch.setattr(workflow, "retrieve_jira", fake_retrieve_jira)
    monkeypatch.setattr(
        workflow,
        "generate_cited_answer",
        fake_generate_answer,
    )

    result = asyncio.run(workflow.run_workflow("  original question  "))

    assert result.original_question == "original question"
    assert result.subquestions == ["GitHub question", "cross-source question"]
    assert result.final_answer == "Grounded answer."
    assert github_questions == ["GitHub question", "cross-source question"]
    assert notion_questions == ["cross-source question"]
    assert jira_questions == ["cross-source question"]
    assert gmail_questions == ["cross-source question"]
    assert [item.source for item in result.retrieval_results] == [
        "github",
        "github",
        "gmail",
        "notion",
        "jira",
    ]
    assert [item.status for item in result.retrieval_results] == [
        "success",
        "success",
        "success",
        "success",
        "success",
    ]
    assert answer_input["question"] == "original question"
    assert answer_input["retrieval_results"] == result.retrieval_results

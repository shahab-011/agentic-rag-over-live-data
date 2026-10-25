import asyncio
import base64
from typing import ClassVar

import pytest

from agentic_rag_over_live_data_and_memory import github_retriever


class FakeMcpTool:
    name = "list_commits"
    description = "List commits in a repository."
    args_schema: ClassVar[dict] = {
        "type": "object",
        "required": ["owner", "repo"],
        "properties": {
            "owner": {"type": "string"},
            "repo": {"type": "string"},
            "perPage": {"type": "integer", "minimum": 1, "maximum": 100},
        },
    }

    def __init__(self) -> None:
        self.arguments = None

    async def ainvoke(self, arguments):
        self.arguments = arguments
        return [{"type": "text", "text": "[]"}]


class FakeGetCommitTool(FakeMcpTool):
    name = "get_commit"
    description = "Get details for a single commit."
    args_schema: ClassVar[dict] = {
        "type": "object",
        "required": ["owner", "repo", "sha"],
        "properties": {
            "owner": {"type": "string"},
            "repo": {"type": "string"},
            "sha": {"type": "string"},
        },
    }


class FakePlanner:
    def __init__(self, plan) -> None:
        self.plan = plan
        self.messages = None

    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages):
        self.messages = messages
        return self.plan


def test_personal_repository_count_question_uses_authenticated_profile(
    monkeypatch,
) -> None:
    expected = {
        "source": "github",
        "tool": "get_authenticated_repository_count",
        "repository": "test-user",
        "subquestion": "How many repositories are in my GitHub profile?",
        "evidence": {
            "login": "test-user",
            "total_repositories": 4,
            "public_repositories": 3,
            "private_repositories": 1,
            "html_url": "https://github.com/test-user",
        },
    }

    async def fake_summary():
        return expected["evidence"]

    monkeypatch.setattr(
        github_retriever,
        "get_authenticated_repository_summary",
        fake_summary,
    )
    monkeypatch.setattr(
        github_retriever,
        "get_github_client",
        lambda: (_ for _ in ()).throw(
            AssertionError("MCP repository-activity client should not be used")
        ),
    )

    result = asyncio.run(
        github_retriever.retrieve_github(
            "How many repositories are in my GitHub profile?"
        )
    )

    assert result == expected


def test_personal_repository_name_request_uses_authenticated_profile(
    monkeypatch,
) -> None:
    repositories = [
        {
            "name": "project-one",
            "full_name": "test-user/project-one",
            "html_url": "https://github.com/test-user/project-one",
            "private": False,
        }
    ]

    async def fake_summary(include_repositories=False):
        assert include_repositories is True
        return {
            "login": "test-user",
            "total_repositories": 1,
            "public_repositories": 1,
            "private_repositories": 0,
            "visibility_scope": "Repositories visible to the token.",
            "repositories": repositories,
        }

    monkeypatch.setattr(
        github_retriever,
        "get_authenticated_repository_summary",
        fake_summary,
    )
    monkeypatch.setattr(
        github_retriever,
        "get_github_client",
        lambda: (_ for _ in ()).throw(
            AssertionError("MCP search should not be used for profile repos")
        ),
    )

    result = asyncio.run(
        github_retriever.retrieve_github(
            "Generate all the repos name from my github"
        )
    )

    assert result["tool"] == "get_authenticated_repository_list"
    assert result["repository"] == "test-user"
    assert result["evidence"]["repositories"] == repositories


def test_authenticated_repository_summary_paginates_all_results(
    monkeypatch,
) -> None:
    page_one = [
        {
            "name": f"project-{index}",
            "full_name": f"test-user/project-{index}",
            "html_url": f"https://github.com/test-user/project-{index}",
            "private": False,
        }
        for index in range(100)
    ]
    page_two = [
        {
            "name": f"project-{index}",
            "full_name": f"test-user/project-{index}",
            "html_url": f"https://github.com/test-user/project-{index}",
            "private": index == 100,
        }
        for index in range(100, 102)
    ]
    requested_pages: list[int] = []

    class FakeResponse:
        def __init__(self, payload) -> None:
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self.payload

    class FakeAsyncClient:
        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def get(self, url: str, params=None):
            if url.endswith("/user"):
                return FakeResponse(
                    {"login": "test-user", "html_url": "https://github.com/test-user"}
                )

            page = params["page"]
            requested_pages.append(page)
            return FakeResponse({1: page_one, 2: page_two}[page])

    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setattr(
        github_retriever.httpx,
        "AsyncClient",
        FakeAsyncClient,
    )

    summary = asyncio.run(
        github_retriever.get_authenticated_repository_summary(
            include_repositories=True
        )
    )

    assert requested_pages == [1, 2]
    assert summary["total_repositories"] == 102
    assert summary["public_repositories"] == 101
    assert summary["private_repositories"] == 1
    assert len(summary["repositories"]) == 102
    assert summary["repositories"][-1]["name"] == "project-101"


def test_groq_planner_selects_tool_and_repository_defaults(
    monkeypatch,
) -> None:
    tool = FakeMcpTool()
    planner = FakePlanner(
        github_retriever.GitHubCallPlan(
            tool_name="list_commits",
            arguments={"perPage": 5},
        )
    )
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "test-model")
    monkeypatch.setattr(
        github_retriever,
        "ChatGroq",
        lambda **kwargs: planner,
    )

    selected_tool, arguments = github_retriever.plan_github_call(
        "Show the five most recent commits.",
        [tool],
        "configured-owner",
        "configured-repo",
    )

    assert selected_tool is tool
    assert arguments == {
        "owner": "configured-owner",
        "repo": "configured-repo",
        "perPage": 5,
    }
    assert "precise GitHub MCP call planner" in planner.messages[0][1]
    assert '"list_commits"' in planner.messages[1][1]


def test_groq_planner_rejects_unavailable_tool(monkeypatch) -> None:
    planner = FakePlanner(
        github_retriever.GitHubCallPlan(
            tool_name="delete_repository",
            arguments={},
        )
    )
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "test-model")
    monkeypatch.setattr(
        github_retriever,
        "ChatGroq",
        lambda **kwargs: planner,
    )

    with pytest.raises(ValueError, match="unavailable or non-read-only"):
        github_retriever.plan_github_call(
            "Delete the repository",
            [FakeMcpTool()],
            "configured-owner",
            "configured-repo",
        )


def test_groq_planner_rejects_unsupported_arguments(monkeypatch) -> None:
    planner = FakePlanner(
        github_retriever.GitHubCallPlan(
            tool_name="list_commits",
            arguments={"unexpected": "value"},
        )
    )
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "test-model")
    monkeypatch.setattr(
        github_retriever,
        "ChatGroq",
        lambda **kwargs: planner,
    )

    with pytest.raises(ValueError, match="unsupported arguments"):
        github_retriever.plan_github_call(
            "List commits",
            [FakeMcpTool()],
            "configured-owner",
            "configured-repo",
        )


def test_groq_planner_rejects_missing_required_identifier(
    monkeypatch,
) -> None:
    planner = FakePlanner(
        github_retriever.GitHubCallPlan(
            tool_name="get_commit",
            arguments={},
        )
    )
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "test-model")
    monkeypatch.setattr(
        github_retriever,
        "ChatGroq",
        lambda **kwargs: planner,
    )

    with pytest.raises(ValueError, match="sha"):
        github_retriever.plan_github_call(
            "Show a commit",
            [FakeGetCommitTool()],
            "configured-owner",
            "configured-repo",
        )


def test_groq_planner_rejects_invalid_argument_types(monkeypatch) -> None:
    planner = FakePlanner(
        github_retriever.GitHubCallPlan(
            tool_name="list_commits",
            arguments={"perPage": "five"},
        )
    )
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "test-model")
    monkeypatch.setattr(
        github_retriever,
        "ChatGroq",
        lambda **kwargs: planner,
    )

    with pytest.raises(ValueError, match="must have type"):
        github_retriever.plan_github_call(
            "List commits",
            [FakeMcpTool()],
            "configured-owner",
            "configured-repo",
        )


def test_repository_count_classifier_avoids_other_repository_questions() -> None:
    assert github_retriever.is_authenticated_repository_count_question(
        "How many repositories are in my GitHub profile?"
    )
    assert github_retriever.is_authenticated_repository_count_question(
        "What is the total number of repos I own?"
    )
    assert not github_retriever.is_authenticated_repository_count_question(
        "How many open issues are in the repository?"
    )
    assert not github_retriever.is_authenticated_repository_count_question(
        "Show repositories for the langchain-ai organization."
    )
    assert github_retriever.is_authenticated_repository_list_question(
        "Generate all the repos name from my github"
    )
    assert github_retriever.is_authenticated_repository_list_question(
        "List the repositories in my GitHub account"
    )
    assert not github_retriever.is_authenticated_repository_list_question(
        "How many repositories are in my GitHub profile?"
    )
    assert not github_retriever.is_authenticated_repository_list_question(
        "Show all repositories for the langchain-ai organization."
    )


def test_issue_evidence_gets_citable_github_urls() -> None:
    evidence = {
        "issues": [
            {"number": 42, "title": "Missing URL"},
            {
                "number": 43,
                "title": "Existing URL",
                "html_url": "https://example.com/existing",
            },
        ],
        "totalCount": 2,
    }

    result = github_retriever.add_issue_urls(
        evidence,
        "octo-org",
        "octo-repo",
    )

    assert result["issues"][0]["html_url"] == (
        "https://github.com/octo-org/octo-repo/issues/42"
    )
    assert result["issues"][1]["html_url"] == (
        "https://example.com/existing"
    )


def test_github_file_contents_returns_full_text_file(
    monkeypatch,
) -> None:
    file_content = "first line\n" + ("x" * 500) + "\nlast line\n"
    requested = {}

    class FakeResponse:
        content = b""

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "type": "file",
                "path": "src/a b.py",
                "content": base64.b64encode(file_content.encode()).decode(),
                "encoding": "base64",
                "html_url": "https://github.com/test-user/project/blob/main/src/a%20b.py",
                "sha": "abc123",
            }

    class FakeAsyncClient:
        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def get(self, url: str, params=None, headers=None):
            requested.update(url=url, params=params, headers=headers)
            return FakeResponse()

    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setattr(
        github_retriever.httpx,
        "AsyncClient",
        FakeAsyncClient,
    )

    result = asyncio.run(
        github_retriever.get_github_file_contents(
            "test-user",
            "project",
            "src/a b.py",
            "main",
        )
    )

    assert requested["url"].endswith("/contents/src/a%20b.py")
    assert requested["params"] == {"ref": "main"}
    assert result == {
        "path": "src/a b.py",
        "file_content": file_content,
        "html_url": (
            "https://github.com/test-user/project/blob/main/src/a%20b.py"
        ),
        "sha": "abc123",
    }

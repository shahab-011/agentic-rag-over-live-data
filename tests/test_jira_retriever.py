import asyncio
import json
from pathlib import Path
from typing import ClassVar

import httpx
import pytest

from agentic_rag_over_live_data_and_memory import jira_mcp_server, jira_retriever


def _fake_jira_tools():
    class FakeTool:
        def __init__(self, name, args_schema) -> None:
            self.name = name
            self.args_schema = args_schema

    return [
        FakeTool(
            "search_jira_issues",
            {
                "type": "object",
                "required": ["jql"],
                "properties": {
                    "jql": {"type": "string"},
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
            },
        ),
        FakeTool(
            "list_jira_projects",
            {"type": "object", "properties": {}},
        ),
        FakeTool(
            "get_jira_issue",
            {
                "type": "object",
                "required": ["issue_key"],
                "properties": {"issue_key": {"type": "string"}},
            },
        ),
    ]


def _mock_jira_query_refiner(monkeypatch, plan):
    recorded = {}

    class FakePlanner:
        def invoke(self, messages):
            recorded["messages"] = messages
            return plan

    class FakeChatGroq:
        def __init__(self, **kwargs) -> None:
            recorded["config"] = kwargs

        def with_structured_output(self, schema, *, method):
            recorded["schema"] = schema
            recorded["method"] = method
            return FakePlanner()

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "test-model")
    monkeypatch.setattr(jira_retriever, "ChatGroq", FakeChatGroq)
    return recorded


def test_jira_client_normalizes_site_url_and_passes_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JIRA_URL", "https://example.atlassian.net/path/to/board")
    monkeypatch.setenv("JIRA_USERNAME", "user@example.com")
    monkeypatch.setenv("JIRA_API_KEY", "test-token")
    client_config = {}

    class FakeClient:
        def __init__(self, config) -> None:
            client_config.update(config)

    monkeypatch.setattr(jira_retriever, "MultiServerMCPClient", FakeClient)

    jira_retriever.get_jira_client()

    server_config = client_config["jira"]
    assert server_config["transport"] == "stdio"
    assert server_config["env"] == {
        "JIRA_URL": "https://example.atlassian.net",
        "JIRA_USERNAME": "user@example.com",
        "JIRA_API_KEY": "test-token",
        "PYTHONPATH": str(Path(jira_retriever.__file__).resolve().parents[1]),
    }
    assert server_config["args"] == [
        "-m",
        "agentic_rag_over_live_data_and_memory.jira_mcp_server",
    ]


def test_jira_configuration_requires_site_email_and_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JIRA_URL", "https://example.atlassian.net")
    monkeypatch.setenv("JIRA_USERNAME", "user@example.com")
    monkeypatch.delenv("JIRA_API_KEY", raising=False)

    assert not jira_retriever.is_jira_configured()
    with pytest.raises(ValueError, match="JIRA_API_KEY"):
        jira_retriever.get_jira_client()


def test_retrieve_jira_normalizes_mcp_text_for_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JIRA_URL", "https://example.atlassian.net")
    monkeypatch.setenv("JIRA_USERNAME", "user@example.com")
    monkeypatch.setenv("JIRA_API_KEY", "test-token")

    class FakeTool:
        name = "search_jira_issues"

        async def ainvoke(self, arguments):
            assert arguments == {"jql": "project = ABC"}
            return [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "issues": [
                                {
                                    "key": "ABC-12",
                                    "url": "https://example.atlassian.net/browse/ABC-12",
                                }
                            ]
                        }
                    ),
                }
            ]

    class FakeClient:
        async def get_tools(self):
            return [FakeTool()]

    def fake_plan(question, tools):
        assert question == "Find issues"
        return (
            tools[0],
            {"jql": "project = ABC"},
            "Find Jira issues in project ABC",
        )

    monkeypatch.setattr(
        jira_retriever,
        "get_jira_client",
        lambda: FakeClient(),
    )
    monkeypatch.setattr(jira_retriever, "_plan_jira_call", fake_plan)

    result = asyncio.run(jira_retriever.retrieve_jira("Find issues"))

    assert result["evidence"] == {
        "issues": [
            {
                "key": "ABC-12",
                "url": "https://example.atlassian.net/browse/ABC-12",
            }



        ]
    }


def test_retrieve_jira_refines_question_before_calling_mcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JIRA_URL", "https://example.atlassian.net")
    monkeypatch.setenv("JIRA_USERNAME", "user@example.com")
    monkeypatch.setenv("JIRA_API_KEY", "test-token")
    calls = {}

    class FakeTool:
        name = "search_jira_issues"
        args_schema: ClassVar[dict[str, object]] = {
            "type": "object",
            "required": ["jql"],
            "properties": {
                "jql": {"type": "string"},
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                },
            },
        }

        async def ainvoke(self, arguments):
            calls["tool_arguments"] = arguments
            return [{"type": "text", "text": '{"issues": []}'}]

    class FakeClient:
        async def get_tools(self):
            return [FakeTool()]

    class FakePlanner:
        def invoke(self, messages):
            calls["messages"] = messages
            return jira_retriever.JiraQueryPlan(
                jira_question="Find ECP issues sorted by latest update.",
                operation="search_issues",
                jql="project = ECP ORDER BY updated DESC",
            )

    class FakeChatGroq:
        def __init__(self, **kwargs) -> None:
            calls["llm_config"] = kwargs

        def with_structured_output(self, schema, *, method):
            assert schema is jira_retriever.JiraQueryPlan
            assert method == "json_mode"
            return FakePlanner()

    monkeypatch.setattr(jira_retriever, "get_jira_client", lambda: FakeClient())
    monkeypatch.setattr(jira_retriever, "ChatGroq", FakeChatGroq)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "test-model")

    result = asyncio.run(
        jira_retriever.retrieve_jira(
            "What Jira issues were updated recently in ECP?"
        )
    )

    assert result["tool"] == "search_jira_issues"
    assert result["jira_question"] == "Find ECP issues sorted by latest update."
    assert calls["llm_config"]["temperature"] == 0
    assert calls["llm_config"]["model"] == "test-model"
    assert calls["tool_arguments"] == {
        "jql": "project = ECP ORDER BY updated DESC",
        "max_results": 10,
    }
    assert result["evidence"] == {"issues": []}


def test_jira_server_search_uses_read_only_api_and_returns_issue_links(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "JIRA_URL",
        "https://example.atlassian.net/jira/software/projects/ABC/boards/1",
    )
    monkeypatch.setenv("JIRA_USERNAME", "user@example.com")
    monkeypatch.setenv("JIRA_API_KEY", "test-token")
    requests: list[httpx.Request] = []
    real_client = httpx.Client

    def handle_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "issues": [
                    {
                        "key": "ABC-12",
                        "fields": {
                            "summary": "MCP connector",
                            "status": {"name": "In Progress"},
                            "issuetype": {"name": "Task"},
                            "priority": {"name": "Medium"},
                            "assignee": {"displayName": "Jira User"},
                            "reporter": {"displayName": "Reporter"},
                            "project": {"name": "Example project"},
                            "created": "2026-01-01",
                            "updated": "2026-01-02",
                            "duedate": None,
                            "description": {
                                "content": [
                                    {
                                        "type": "paragraph",
                                        "content": [{"text": "Implement MCP"}],
                                    }
                                ]
                            },
                        },
                    }
                ]
            },
        )

    monkeypatch.setattr(
        jira_mcp_server.httpx,
        "Client",
        lambda **kwargs: real_client(
            transport=httpx.MockTransport(handle_request),
            **kwargs,
        ),
    )

    result = json.loads(
        jira_mcp_server.search_jira_issues(
            "project = ABC ORDER BY updated DESC",
            max_results=1,
        )
    )

    assert requests[0].method == "GET"
    assert requests[0].url.path == "/rest/api/3/search/jql"
    assert requests[0].url.params["jql"] == "project = ABC ORDER BY updated DESC"
    assert requests[0].headers["Authorization"].startswith("Basic ")
    assert result["issues"][0] == {
        "key": "ABC-12",
        "url": "https://example.atlassian.net/browse/ABC-12",
        "summary": "MCP connector",
        "status": "In Progress",
        "issue_type": "Task",
        "priority": "Medium",
        "assignee": "Jira User",
        "reporter": "Reporter",
        "project": "Example project",
        "created": "2026-01-01",
        "updated": "2026-01-02",
        "due_date": None,
        "description": "Implement MCP",
    }


def test_jira_server_rejects_invalid_search_limits() -> None:
    with pytest.raises(ValueError, match="max_results"):
        jira_mcp_server.search_jira_issues("project = ABC", max_results=100)


def test_jira_ai_refiner_creates_jql_for_recent_project_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = "What Jira issues were updated recently in ECP?"
    plan = jira_retriever.JiraQueryPlan(
        jira_question="Find ECP issues sorted by most recently updated.",
        operation="search_issues",
        jql="project = ECP ORDER BY updated DESC",
    )
    recorded = _mock_jira_query_refiner(monkeypatch, plan)

    tool, arguments, jira_question = jira_retriever._plan_jira_call(
        question,
        _fake_jira_tools(),
    )

    assert tool.name == "search_jira_issues"
    assert arguments == {
        "jql": "project = ECP ORDER BY updated DESC",
        "max_results": 10,
    }
    assert jira_question == "Find ECP issues sorted by most recently updated."
    assert recorded["schema"] is jira_retriever.JiraQueryPlan
    assert recorded["method"] == "json_mode"
    assert recorded["config"]["temperature"] == 0
    assert "unless the user" in recorded["messages"][0][1]
    assert question in recorded["messages"][1][1]


def test_jira_query_plan_accepts_nested_groq_operation_shape() -> None:
    plan = jira_retriever.JiraQueryPlan.model_validate(
        {
            "question": (
                "Search for issues in the ECP project sorted by most recent update"
            ),
            "operation": {
                "type": "search_issues",
                "jql": "project = ECP ORDER BY updated DESC",
                "issue_key": "",
            },
        }
    )

    assert plan.jira_question.startswith("Search for issues in the ECP project")
    assert plan.operation == "search_issues"
    assert plan.jql == "project = ECP ORDER BY updated DESC"
    assert plan.issue_key == ""


def test_jira_ai_refiner_selects_project_list_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_jira_query_refiner(
        monkeypatch,
        jira_retriever.JiraQueryPlan(
            jira_question="List projects available in Jira.",
            operation="list_projects",
        ),
    )

    tool, arguments, jira_question = jira_retriever._plan_jira_call(
        "List the Jira projects I can access.",
        _fake_jira_tools(),
    )

    assert tool.name == "list_jira_projects"
    assert arguments == {}
    assert jira_question == "List projects available in Jira."


def test_jira_ai_refiner_looks_up_only_an_explicit_issue_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_jira_query_refiner(
        monkeypatch,
        jira_retriever.JiraQueryPlan(
            jira_question="Get the status of issue ECP-123.",
            operation="get_issue",
            issue_key="ECP-123",
        ),
    )

    tool, arguments, _ = jira_retriever._plan_jira_call(
        "What is the status of issue ECP-123?",
        _fake_jira_tools(),
    )

    assert tool.name == "get_jira_issue"
    assert arguments == {"issue_key": "ECP-123"}


def test_jira_ai_refiner_rejects_invented_project_or_issue_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = _fake_jira_tools()
    _mock_jira_query_refiner(
        monkeypatch,
        jira_retriever.JiraQueryPlan(
            jira_question="Find recently updated project ABC issues.",
            operation="search_issues",
            jql="project = ABC ORDER BY updated DESC",
        ),
    )
    with pytest.raises(ValueError, match="project key"):
        jira_retriever._plan_jira_call(
            "What Jira issues were updated recently?",
            tools,
        )

    _mock_jira_query_refiner(
        monkeypatch,
        jira_retriever.JiraQueryPlan(
            jira_question="Find issues in ECP and ABC.",
            operation="search_issues",
            jql="project in (ECP, ABC) ORDER BY updated DESC",
        ),
    )
    with pytest.raises(ValueError, match="project key"):
        jira_retriever._plan_jira_call(
            "What Jira issues were updated recently in ECP?",
            tools,
        )

    _mock_jira_query_refiner(
        monkeypatch,
        jira_retriever.JiraQueryPlan(
            jira_question="Get issue ECP-999.",
            operation="get_issue",
            issue_key="ECP-999",
        ),
    )
    with pytest.raises(ValueError, match="explicitly present"):
        jira_retriever._plan_jira_call(
            "What is the status of the latest ECP issue?",
            tools,
        )


def test_jira_ai_refiner_rejects_write_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_jira_query_refiner(
        monkeypatch,
        jira_retriever.JiraQueryPlan(
            jira_question="Delete issue ECP-123.",
            operation="unsupported",
        ),
    )

    with pytest.raises(ValueError, match="read-only"):
        jira_retriever._plan_jira_call(
            "Delete issue ECP-123",
            _fake_jira_tools(),
        )


def test_jira_ai_refiner_rejects_an_invented_recent_time_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_jira_query_refiner(
        monkeypatch,
        jira_retriever.JiraQueryPlan(
            jira_question="Find recently updated ECP issues.",
            operation="search_issues",
            jql="project = ECP AND updated >= -7d ORDER BY updated DESC",
        ),
    )

    with pytest.raises(ValueError, match="date range"):
        jira_retriever._plan_jira_call(
            "What Jira issues were updated recently in ECP?",
            _fake_jira_tools(),
        )


def test_jira_tool_arguments_are_validated_against_mcp_json_schema() -> None:
    schema = {
        "type": "object",
        "required": ["jql"],
        "properties": {
            "jql": {"type": "string"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 50},
        },
    }
    jira_retriever._validate_jira_arguments(
        "search_jira_issues",
        schema,
        {"jql": "project = ABC", "max_results": 10},
    )

    with pytest.raises(ValueError, match="type"):
        jira_retriever._validate_jira_arguments(
            "search_jira_issues",
            schema,
            {"jql": "project = ABC", "max_results": "10"},
        )

    with pytest.raises(ValueError, match="unsupported arguments"):
        jira_retriever._validate_jira_arguments(
            "search_jira_issues",
            schema,
            {"jql": "project = ABC", "extra": "not allowed"},
        )

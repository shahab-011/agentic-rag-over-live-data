
import asyncio
import base64
import json
import os
import re
from typing import Any
from urllib.parse import quote

import httpx
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import AliasChoices, BaseModel, Field

load_dotenv()

_REPOSITORY_WORDS = re.compile(r"\b(repo|repos|repository|repositories)\b")
_COUNT_WORDS = re.compile(r"\b(how many|number of|count|total)\b")

READ_ONLY_GITHUB_TOOLS = frozenset(
    {
        "get_commit",
        "get_file_contents",
        "get_label",
        "get_latest_release",
        "get_release_by_tag",
        "get_tag",
        "issue_read",
        "list_branches",
        "list_commits",
        "list_issue_fields",
        "list_issue_types",
        "list_issues",
        "list_pull_requests",
        "list_releases",
        "list_repository_collaborators",
        "list_tags",
        "pull_request_read",
        "search_code",
        "search_commits",
        "search_issues",
        "search_pull_requests",
        "search_repositories",
    }
)

GITHUB_CALL_PLANNER_PROMPT = """
You are a precise GitHub MCP call planner. Convert the supplied GitHub
subquestion into exactly one read-only call from the available tool catalog.
Return JSON with exactly these fields:
{"tool_name": "<exact catalog tool name>", "arguments": {...}}

Rules:
1. Select only a tool whose exact name appears in the catalog. Never invent
   tools, endpoints, arguments, repository names, IDs, dates, branches, or
   search terms not supported by the subquestion or the provided defaults.
2. Follow the chosen tool's input schema exactly: include every required
   argument, use only listed argument names, obey types and enum values, and
   use GitHub search syntax only for search tools.
3. For repository-scoped tools, use an owner or repository explicitly named
   in the subquestion. Otherwise use the provided default owner/repository.
   Do not apply those defaults to search tools unless their schemas support
   owner/repo and the question is scoped to that repository.
4. Distinguish list operations from search operations. Use search tools when
   the user asks for matching/filtering terms; use list tools for recent or
   unfiltered items. Use get/read tools when the question identifies one
   specific commit, issue, pull request, file, tag, or release.
5. Interpret time ranges and status filters only when stated. Do not assume
   open/closed/all if the selected tool schema provides a suitable filter;
   use the user's requested scope.
6. For code searches, include repository/user/org qualifiers when the
   subquestion specifies scope. For repository search, use GitHub query
   qualifiers such as user:NAME, org:NAME, language:python, or topic:NAME
   when they match the user's request.
7. Do not answer the question. Do not execute calls. Do not include
   commentary, markdown fences, or fields outside the JSON object.
8. If the subquestion cannot be answered by any listed tool or lacks an
   essential identifier, choose the closest safe read-only tool only when
   it can return useful evidence without guessing; otherwise return
   {"tool_name": "", "arguments": {}}.
"""


class GitHubCallPlan(BaseModel):
    tool_name: str = Field(
        validation_alias=AliasChoices("tool_name", "tool", "name")
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("arguments", "args", "parameters"),
    )


def is_authenticated_repository_count_question(question: str) -> bool:
    """Identify questions asking for the authenticated user's repository count."""

    normalized = question.lower()
    has_repository = bool(_REPOSITORY_WORDS.search(normalized))
    has_count = bool(_COUNT_WORDS.search(normalized))
    has_profile_context = any(
        phrase in normalized
        for phrase in (
            "my",
            "mine",
            "i own",
            "i have",
            "my profile",
            "my account",
            "github profile",
        )
    )
    return has_repository and has_count and has_profile_context


def is_authenticated_repository_list_question(question: str) -> bool:
    """Identify requests for repository names in the authenticated account."""

    normalized = question.lower()
    has_repository = bool(_REPOSITORY_WORDS.search(normalized))
    has_name_or_list_intent = bool(
        re.search(
            r"\b(all|list|show|display|generate|name|names)\b",
            normalized,
        )
    )
    has_profile_context = any(
        phrase in normalized
        for phrase in (
            "my",
            "mine",
            "i own",
            "i have",
            "my profile",
            "my account",
            "github profile",
            "github account",
        )
    )
    return (
        has_repository
        and has_name_or_list_intent
        and has_profile_context
        and not is_authenticated_repository_count_question(question)
    )


def get_github_client() -> MultiServerMCPClient:
    """Create a read-only GitHub MCP client."""

    token = os.getenv("GITHUB_TOKEN")

    if not token:
        raise ValueError("GITHUB_TOKEN is missing from .env")

    return MultiServerMCPClient(
        {
            "github": {
                "transport": "http",
                "url": "https://api.githubcopilot.com/mcp/",
                "headers": {
                    "Authorization": f"Bearer {token}",
                    "X-MCP-Toolsets": "repos,issues,pull_requests",
                    "X-MCP-Readonly": "true",
                },
            }
        }
    )


async def get_authenticated_repository_summary(
    include_repositories: bool = False,
) -> dict[str, Any]:
    """Fetch repositories owned by the authenticated account and summarize them."""

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN is missing from .env")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async with httpx.AsyncClient(
        headers=headers,
        timeout=30,
    ) as client:
        profile_response = await client.get("https://api.github.com/user")
        profile_response.raise_for_status()
        profile = profile_response.json()

        login = profile.get("login")
        if not isinstance(login, str) or not login:
            raise RuntimeError(
                "GitHub did not return a login for the authenticated account."
            )

        repositories: list[dict[str, Any]] = []
        page = 1
        while True:
            response = await client.get(
                "https://api.github.com/user/repos",
                params={
                    "visibility": "all",
                    "affiliation": "owner",
                    "per_page": 100,
                    "page": page,
                },
            )
            response.raise_for_status()
            batch = response.json()
            if not isinstance(batch, list):
                raise TypeError(
                    "GitHub returned an unexpected repository-list response."
                )

            repositories.extend(
                {
                    "name": item.get("name"),
                    "full_name": item.get("full_name"),
                    "html_url": item.get("html_url"),
                    "private": item.get("private", False),
                }
                for item in batch
                if isinstance(item, dict)
            )
            if len(batch) < 100:
                break
            page += 1

    public_count = sum(
        not repository.get("private", False)
        for repository in repositories
    )
    private_count = len(repositories) - public_count

    summary = {
        "login": login,
        "total_repositories": len(repositories),
        "public_repositories": public_count,
        "private_repositories": private_count,
        "visibility_scope": (
            "Repositories owned by the authenticated account and visible "
            "to this token."
        ),
    }
    if include_repositories:
        summary["repositories"] = repositories
    else:
        summary["html_url"] = profile.get(
            "html_url",
            f"https://github.com/{login}",
        )
    return summary


async def get_github_file_contents(
    owner: str,
    repo: str,
    path: str,
    ref: str | None = None,
) -> dict[str, Any] | None:
    """Fetch a complete text file from GitHub's Contents API."""

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN is missing from .env")

    if not path or path == "/":
        return None

    encoded_owner = quote(owner, safe="")
    encoded_repo = quote(repo, safe="")
    encoded_path = quote(path.strip("/"), safe="/")
    url = (
        f"https://api.github.com/repos/{encoded_owner}/{encoded_repo}"
        f"/contents/{encoded_path}"
    )
    params = {"ref": ref} if ref else None
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async with httpx.AsyncClient(
        headers=headers,
        timeout=30,
    ) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()

        if not isinstance(payload, dict) or payload.get("type") != "file":
            return None

        content = payload.get("content")
        encoding = payload.get("encoding")
        if encoding == "base64" and isinstance(content, str):
            file_bytes = base64.b64decode(content)
        else:
            raw_response = await client.get(
                url,
                params=params,
                headers={"Accept": "application/vnd.github.raw+json"},
            )
            raw_response.raise_for_status()
            file_bytes = raw_response.content

    try:
        file_content = file_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{path} is not a UTF-8 text file and cannot be displayed as code."
        ) from exc

    html_url = payload.get("html_url")
    if not isinstance(html_url, str) or not html_url:
        raise RuntimeError("GitHub did not return a link for the requested file.")

    return {
        "path": payload.get("path", path),
        "file_content": file_content,
        "html_url": html_url,
        "sha": payload.get("sha"),
    }


def _compact_tool_schema(tool: Any) -> dict[str, Any]:
    schema = tool.args_schema
    if not isinstance(schema, dict):
        raise TypeError(f"GitHub tool {tool.name} has no JSON input schema.")

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise TypeError(f"GitHub tool {tool.name} has an invalid input schema.")

    compact_properties: dict[str, Any] = {}
    for name, property_schema in properties.items():
        if not isinstance(property_schema, dict):
            continue
        compact_property = {
            key: property_schema[key]
            for key in (
                "type",
                "description",
                "enum",
                "default",
                "minimum",
                "maximum",
                "items",
                "required",
                "properties",
            )
            if key in property_schema
        }
        if "description" in compact_property:
            compact_property["description"] = str(
                compact_property["description"]
            )[:180]
        compact_properties[name] = compact_property

    return {
        "name": tool.name,
        "description": str(tool.description or "")[:220],
        "required": schema.get("required", []),
        "properties": compact_properties,
    }


def _matches_json_type(value: Any, expected_type: Any) -> bool:
    if isinstance(expected_type, list):
        return any(_matches_json_type(value, item) for item in expected_type)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "null":
        return value is None
    return True


def _validate_argument_value(
    name: str,
    value: Any,
    schema: dict[str, Any],
) -> None:
    expected_type = schema.get("type")
    if expected_type is not None and not _matches_json_type(
        value,
        expected_type,
    ):
        raise ValueError(
            f"GitHub argument {name!r} must have type {expected_type!r}."
        )

    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(
            f"GitHub argument {name!r} must be one of {schema['enum']!r}."
        )

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(
                f"GitHub argument {name!r} is below its minimum."
            )
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(
                f"GitHub argument {name!r} is above its maximum."
            )

    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for item in value:
            _validate_argument_value(
                name,
                item,
                schema["items"],
            )


def _validate_github_call(
    tool: Any,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    schema = tool.args_schema
    if not isinstance(schema, dict):
        raise TypeError(f"GitHub tool {tool.name} has no JSON input schema.")

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise TypeError(f"GitHub tool {tool.name} has an invalid input schema.")

    unknown_arguments = set(arguments) - set(properties)
    if unknown_arguments:
        raise ValueError(
            f"Planner returned unsupported arguments for {tool.name}: "
            f"{', '.join(sorted(unknown_arguments))}."
        )

    required = schema.get("required", [])
    missing_arguments = set(required) - set(arguments)
    if missing_arguments:
        raise ValueError(
            f"Planner omitted required arguments for {tool.name}: "
            f"{', '.join(sorted(missing_arguments))}."
        )

    for name, value in arguments.items():
        property_schema = properties.get(name)
        if isinstance(property_schema, dict):
            _validate_argument_value(name, value, property_schema)

    return arguments


def _listing_tool_arguments(tool: Any, owner: str, repo: str) -> dict[str, Any] | None:
    schema = tool.args_schema if isinstance(tool.args_schema, dict) else {}
    properties = schema.get("properties", {})
    arguments: dict[str, Any] = {}
    if "owner" in properties:
        arguments["owner"] = owner
    if "repo" in properties:
        arguments["repo"] = repo

    for name in schema.get("required", []):
        if name in arguments:
            continue
        property_schema = properties.get(name, {})
        if "default" in property_schema:
            arguments[name] = property_schema["default"]
        elif name == "state" and "all" in property_schema.get("enum", []):
            arguments[name] = "all"
        else:
            return None
    return _validate_github_call(tool, arguments)


def _is_broad_activity_question(question: str) -> bool:
    normalized = question.lower()
    return (
        bool(re.search(r"\b(recent|latest|activity)\b", normalized))
        and "commit" in normalized
        and ("pull request" in normalized or "pull requests" in normalized)
        and "issue" in normalized
    )


async def _retrieve_github_activity(
    tools: list[Any],
    owner: str,
    repo: str,
) -> dict[str, Any] | None:
    tool_map = {tool.name: tool for tool in tools if tool.name in READ_ONLY_GITHUB_TOOLS}
    categories = (
        ("commits", "list_commits"),
        ("pull_requests", "list_pull_requests"),
        ("issues", "list_issues"),
    )
    evidence: dict[str, Any] = {}
    for category, tool_name in categories:
        tool = tool_map.get(tool_name)
        if tool is None:
            continue
        arguments = _listing_tool_arguments(tool, owner, repo)
        if arguments is None:
            continue
        result = await tool.ainvoke(arguments)
        normalized = normalize_mcp_result(result)
        if isinstance(normalized, str) and normalized.lstrip().startswith("Error executing tool"):
            raise RuntimeError(normalized)
        if tool_name == "list_issues":
            normalized = add_issue_urls(normalized, owner, repo)
        evidence[category] = normalized

    if not evidence:
        return None
    return evidence


def plan_github_call(
    subquestion: str,
    tools: list[Any],
    owner: str,
    repo: str,
) -> tuple[Any, dict[str, Any]]:
    """Ask Groq to map a subquestion to one validated read-only MCP call."""

    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from .env.")
    if not model:
        raise ValueError("GROQ_MODEL is missing from .env.")

    available_tools = [
        tool
        for tool in tools
        if tool.name in READ_ONLY_GITHUB_TOOLS
    ]
    if not available_tools:
        raise RuntimeError("No supported read-only GitHub MCP tools are available.")

    tool_map = {tool.name: tool for tool in available_tools}
    catalog = [
        _compact_tool_schema(tool)
        for tool in available_tools
    ]

    llm = ChatGroq(
        model=model,
        api_key=api_key,
        temperature=0,
    )
    planner = llm.with_structured_output(
        GitHubCallPlan,
        method="json_mode",
    )
    plan = planner.invoke(
        [
            ("system", GITHUB_CALL_PLANNER_PROMPT),
            (
                "human",
                "Subquestion:\n"
                f"{subquestion}\n\n"
                "Default repository context (use only if needed):\n"
                f"owner={owner}\nrepo={repo}\n\n"
                "Available read-only GitHub MCP tools and input schemas:\n"
                + json.dumps(catalog, ensure_ascii=True),
            ),
        ]
    )

    tool = tool_map.get(plan.tool_name)
    if tool is None:
        raise ValueError(
            f"Groq selected an unavailable or non-read-only GitHub tool: "
            f"{plan.tool_name!r}."
        )

    arguments = dict(plan.arguments)
    schema = tool.args_schema
    required = schema.get("required", []) if isinstance(schema, dict) else []
    if "owner" in required:
        arguments.setdefault("owner", owner)
    if "repo" in required:
        arguments.setdefault("repo", repo)

    return tool, _validate_github_call(tool, arguments)


def normalize_mcp_result(result: Any) -> Any:
    """
    Convert JSON strings in MCP text blocks into Python objects.

    For example:
        [{"type": "text", "text": "[{\\"number\\": 1}]"}]

    becomes:
        [{"number": 1}]
    """

    if isinstance(result, list):
        normalized = []

        for item in result:
            if (
                isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ):
                text = item["text"]

                try:
                    parsed = json.loads(text)
                    normalized.append(normalize_mcp_result(parsed))
                except json.JSONDecodeError:
                    # Preserve plain text when it is not JSON.
                    normalized.append(text)

            else:
                normalized.append(normalize_mcp_result(item))

        # Unwrap one text block to expose its contents directly.
        if len(normalized) == 1:
            return normalized[0]

        return normalized

    if isinstance(result, dict):
        return {
            key: normalize_mcp_result(value)
            for key, value in result.items()
        }

    return result


def add_issue_urls(
    result: Any,
    owner: str,
    repo: str,
) -> Any:
    """Add deterministic GitHub issue links when MCP omits them from results."""

    if isinstance(result, list):
        for item in result:
            add_issue_urls(item, owner, repo)
    elif isinstance(result, dict):
        issues = result.get("issues")
        if isinstance(issues, list):
            for issue in issues:
                if isinstance(issue, dict):
                    number = issue.get("number")
                    if isinstance(number, int) and not isinstance(number, bool):
                        issue.setdefault(
                            "html_url",
                            f"https://github.com/{owner}/{repo}/issues/{number}",
                        )
        for value in result.values():
            if isinstance(value, (dict, list)):
                add_issue_urls(value, owner, repo)
    return result


async def retrieve_github(
    subquestion: str,
    owner: str | None = None,
    repo: str | None = None,
) -> dict[str, Any]:
    """Retrieve and normalize GitHub evidence for a subquestion."""

    owner = owner or os.getenv("GITHUB_OWNER", "langchain-ai")
    repo = repo or os.getenv("GITHUB_REPO", "langgraph")

    if is_authenticated_repository_list_question(subquestion):
        evidence = await get_authenticated_repository_summary(
            include_repositories=True,
        )
        return {
            "source": "github",
            "tool": "get_authenticated_repository_list",
            "repository": evidence["login"],
            "subquestion": subquestion,
            "evidence": evidence,
        }

    if is_authenticated_repository_count_question(subquestion):
        evidence = await get_authenticated_repository_summary()
        return {
            "source": "github",
            "tool": "get_authenticated_repository_count",
            "repository": evidence["login"],
            "subquestion": subquestion,
            "evidence": evidence,
        }

    client = get_github_client()
    tools = await client.get_tools()
    if _is_broad_activity_question(subquestion):
        activity_evidence = await _retrieve_github_activity(tools, owner, repo)
        if activity_evidence is not None:
            return {
                "source": "github",
                "tool": "list_github_activity",
                "repository": f"{owner}/{repo}",
                "subquestion": subquestion,
                "evidence": activity_evidence,
            }

    try:
        tool, arguments = plan_github_call(
            subquestion,
            tools,
            owner,
            repo,
        )
    except ValueError as exc:
        if "unavailable or non-read-only GitHub tool" not in str(exc):
            raise
        fallback_names = []
        normalized_question = subquestion.lower()
        if "pull request" in normalized_question:
            fallback_names.append("list_pull_requests")
        if "commit" in normalized_question:
            fallback_names.append("list_commits")
        if "issue" in normalized_question:
            fallback_names.append("list_issues")
        fallback = next(
            (
                (tool, _listing_tool_arguments(tool, owner, repo))
                for name in fallback_names
                for tool in tools
                if tool.name == name
                and _listing_tool_arguments(tool, owner, repo) is not None
            ),
            None,
        )
        if fallback is None:
            raise
        tool, arguments = fallback

    if tool.name == "get_file_contents":
        structured_evidence = await get_github_file_contents(
            owner=arguments.get("owner", owner),
            repo=arguments.get("repo", repo),
            path=arguments.get("path", "/"),
            ref=arguments.get("ref") or arguments.get("sha"),
        )
    else:
        structured_evidence = None

    if structured_evidence is None:
        # Execute the selected MCP tool.
        raw_result = await tool.ainvoke(arguments)
        structured_evidence = normalize_mcp_result(raw_result)

    if tool.name == "list_issues":
        structured_evidence = add_issue_urls(
            structured_evidence,
            owner,
            repo,
        )

    return {
        "source": "github",
        "tool": tool.name,
        "repository": f"{owner}/{repo}",
        "subquestion": subquestion,
        "evidence": structured_evidence,
    }


async def main() -> None:
    """Run the GitHub retriever independently for testing."""

    question = input(
        "What would you like to retrieve from GitHub? "
    ).strip()

    if not question:
        print("Please enter a question.")
        return

    try:
        result = await retrieve_github(question)

        print("\n========== GITHUB RETRIEVAL RESULT ==========\n")
        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )

    except (RuntimeError, ValueError) as exc:
        print(f"\nGitHub retrieval failed: {exc}")


if __name__ == "__main__":
    asyncio.run(main())

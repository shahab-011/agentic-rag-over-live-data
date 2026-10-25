import json
import os
from typing import Any
from urllib.parse import quote, urlparse

import httpx
from mcp.server.fastmcp import FastMCP

server = FastMCP("jira")

ISSUE_FIELDS = (
    "summary,status,assignee,reporter,priority,issuetype,project,"
    "created,updated,duedate,description"
)


def _jira_base_url() -> str:
    value = os.getenv("JIRA_URL", "").strip()
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("JIRA_URL must be an HTTPS Jira site URL.")
    return f"{parsed.scheme}://{parsed.netloc}"


def _jira_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    email = os.getenv("JIRA_USERNAME", "").strip()
    token = os.getenv("JIRA_API_KEY", "").strip()
    if not email or not token:
        raise ValueError("JIRA_USERNAME and JIRA_API_KEY are required.")

    with httpx.Client(
        base_url=_jira_base_url(),
        auth=(email, token),
        headers={"Accept": "application/json"},
        timeout=30,
    ) as client:
        response = client.get(path, params=params)
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError("Jira returned an unexpected response.")
    return payload


def _plain_text(value: Any) -> str:
    fragments: list[str] = []

    def collect(item: Any) -> None:
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                fragments.append(text)
            for child in item.values():
                if isinstance(child, (dict, list)):
                    collect(child)
        elif isinstance(item, list):
            for child in item:
                collect(child)

    collect(value)
    return " ".join(" ".join(fragments).split())[:4000]


def _issue_summary(issue: Any) -> dict[str, Any] | None:
    if not isinstance(issue, dict):
        return None
    key = issue.get("key")
    fields = issue.get("fields")
    if not isinstance(key, str) or not isinstance(fields, dict):
        return None

    def field_name(field: str, property_name: str = "name") -> str | None:
        value = fields.get(field)
        name = value.get(property_name) if isinstance(value, dict) else None
        return name if isinstance(name, str) else None

    assignee = fields.get("assignee")
    reporter = fields.get("reporter")
    project = fields.get("project")
    return {
        "key": key,
        "url": f"{_jira_base_url()}/browse/{quote(key, safe='')}",
        "summary": fields.get("summary"),
        "status": field_name("status"),
        "issue_type": field_name("issuetype"),
        "priority": field_name("priority"),
        "assignee": (
            assignee.get("displayName")
            if isinstance(assignee, dict)
            else None
        ),
        "reporter": (
            reporter.get("displayName")
            if isinstance(reporter, dict)
            else None
        ),
        "project": (
            project.get("name") if isinstance(project, dict) else None
        ),
        "created": fields.get("created"),
        "updated": fields.get("updated"),
        "due_date": fields.get("duedate"),
        "description": _plain_text(fields.get("description")),
    }


@server.tool()
def search_jira_issues(jql: str, max_results: int = 10) -> str:
    """Search Jira issues using JQL. This operation does not modify Jira."""
    if not jql.strip():
        raise ValueError("A JQL query is required.")
    if not 1 <= max_results <= 50:
        raise ValueError("max_results must be between 1 and 50.")

    payload = _jira_get(
        "/rest/api/3/search/jql",
        params={
            "jql": jql,
            "maxResults": max_results,
            "fields": ISSUE_FIELDS,
        },
    )
    issues = payload.get("issues")
    if not isinstance(issues, list):
        raise TypeError("Jira returned an invalid issue list.")
    return json.dumps(
        {
            "issues": [
                summary
                for issue in issues
                if (summary := _issue_summary(issue)) is not None
            ]
        },
        ensure_ascii=False,
    )


@server.tool()
def get_jira_issue(issue_key: str) -> str:
    """Get one Jira issue by key. This operation does not modify Jira."""
    if not issue_key.strip():
        raise ValueError("An issue key is required.")
    payload = _jira_get(
        f"/rest/api/3/issue/{quote(issue_key.strip(), safe='')}",
        params={"fields": ISSUE_FIELDS},
    )
    issue = _issue_summary(payload)
    if issue is None:
        raise TypeError("Jira returned an invalid issue.")
    return json.dumps({"issue": issue}, ensure_ascii=False)


@server.tool()
def list_jira_projects() -> str:
    """List Jira projects visible to the authenticated account."""
    payload = _jira_get(
        "/rest/api/3/project/search",
        params={"maxResults": 50},
    )
    projects = payload.get("values")
    if not isinstance(projects, list):
        raise TypeError("Jira returned an invalid project list.")
    return json.dumps(
        {
            "projects": [
                {
                    "key": project["key"],
                    "name": project["name"],
                    "url": f"{_jira_base_url()}/browse/{quote(project['key'], safe='')}",
                }
                for project in projects
                if isinstance(project, dict)
                and isinstance(project.get("key"), str)
                and isinstance(project.get("name"), str)
            ]
        },
        ensure_ascii=False,
    )


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()

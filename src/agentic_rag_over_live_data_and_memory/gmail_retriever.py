import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

READ_ONLY_GMAIL_TOOLS = frozenset({"search_emails", "get_email", "list_labels"})
READ_ONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
_QUOTED_PHRASE = re.compile(r"[\"“‘'](.+?)[\"”’']")
_SUBJECT_PHRASE = re.compile(
    r"\bsubject\s*(?:is\s+|:\s*)?[\"“‘'](.+?)[\"”’']",
    re.IGNORECASE,
)
_SENDER_PHRASE = re.compile(
    r"\bfrom\s+([\w.@+-]+(?:\s+(?!and\b|with\b|about\b|on\b|that\b|to\b)[\w.@+-]+){0,2})",
    re.IGNORECASE,
)
_PROJECT_EXPANSIONS = {"ECP": "Enterprise Cloud Platform"}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _oauth_paths() -> tuple[Path, Path]:
    client_path = Path(os.getenv("GMAIL_OAUTH_CLIENT_PATH", "credentials/gmail-oauth-client.json")).expanduser()
    token_path = Path(os.getenv("GMAIL_OAUTH_TOKEN_PATH", "credentials/gmail-token.json")).expanduser()
    if not client_path.is_absolute():
        client_path = _project_root() / client_path
    if not token_path.is_absolute():
        token_path = _project_root() / token_path
    return client_path, token_path


def is_gmail_configured() -> bool:
    client_path, _ = _oauth_paths()
    return client_path.is_file()


def is_gmail_authenticated() -> bool:
    _, token_path = _oauth_paths()
    if not token_path.is_file():
        return False

    try:
        # Load the scopes recorded in the token file. Passing a requested scope
        # here would make the library report that scope without verifying it was
        # actually granted by Google.
        credentials = Credentials.from_authorized_user_file(str(token_path))
    except (OSError, ValueError, json.JSONDecodeError):
        return False

    granted_scopes = set(credentials.scopes or ())
    has_read_scope = READ_ONLY_SCOPE in granted_scopes
    usable_or_refreshable = credentials.valid or (
        credentials.expired and bool(credentials.refresh_token)
    )
    return bool(credentials.token and has_read_scope and usable_or_refreshable)


def get_gmail_client() -> MultiServerMCPClient:
    client_path, token_path = _oauth_paths()
    if not client_path.is_file():
        raise ValueError("GMAIL_OAUTH_CLIENT_PATH does not point to an existing OAuth client JSON file.")

    return MultiServerMCPClient({
        "gmail": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "agentic_rag_over_live_data_and_memory.gmail_mcp_server"],
            "env": {
                "GMAIL_OAUTH_CLIENT_PATH": str(client_path),
                "GMAIL_OAUTH_TOKEN_PATH": str(token_path),
                "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            },
        }
    })


def _normalize_mcp_result(result: Any) -> Any:
    if isinstance(result, str):
        try:
            return _normalize_mcp_result(json.loads(result))
        except json.JSONDecodeError:
            return result
    if isinstance(result, list):
        normalized = [_normalize_mcp_result(item) for item in result]
        if len(normalized) == 1 and isinstance(result[0], dict) and result[0].get("type") == "text":
            return normalized[0]
        return normalized
    if isinstance(result, dict):
        if result.get("type") == "text" and isinstance(result.get("text"), str):
            return _normalize_mcp_result(result["text"])
        return {key: _normalize_mcp_result(value) for key, value in result.items()}
    return result


def _validate_arguments(tool: Any, arguments: dict[str, Any]) -> dict[str, Any]:
    schema = tool.args_schema if isinstance(tool.args_schema, dict) else {}
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    unknown = set(arguments) - set(properties)
    if unknown:
        raise ValueError(f"Unsupported Gmail MCP arguments: {sorted(unknown)}")
    required = schema.get("required", [])
    missing = set(required) - set(arguments)
    if missing:
        raise ValueError(f"Missing Gmail MCP arguments: {sorted(missing)}")
    return arguments


def _gmail_search_query(question: str) -> tuple[str, str | None]:
    """Convert common natural-language email requests into Gmail search syntax."""
    subject_match = _SUBJECT_PHRASE.search(question)
    quoted_match = _QUOTED_PHRASE.search(question)
    subject = subject_match.group(1).strip() if subject_match else None
    phrase = subject or (quoted_match.group(1).strip() if quoted_match else None)

    sender_match = _SENDER_PHRASE.search(question)
    sender = sender_match.group(1).strip() if sender_match else None

    if subject:
        query = f'subject:"{subject}"'
    elif phrase:
        query = f'"{phrase}"'
    else:
        # Strip conversational filler that Gmail would otherwise treat as
        # required search terms (for example, "show me the full email").
        query = re.sub(
            r"\b(show|find|read|fetch|full|entire|complete|email|emails|message|messages|"
            r"in|my|gmail|with|the|about|please|me|from|subject|titled|called)\b",
            " ",
            question,
            flags=re.IGNORECASE,
        )
        query = " ".join(query.split()).strip(" ?.,:;") or question

    if sender:
        query = f"{query} from:{sender}"
    return query, sender if subject else None


def _gmail_search_queries(question: str) -> list[str]:
    """Prefer broad project identifiers over conversational filler words."""
    query, sender = _gmail_search_query(question)
    if query.startswith("subject:"):
        queries = [query]
        if sender:
            queries.append(query.rsplit(" from:", 1)[0])
        return queries

    project_codes = re.findall(r"\b[A-Z]{2,10}(?:-\d+)?\b", question)
    project_code = next(
        (
            re.sub(r"-\d+$", "", code)
            for code in project_codes
            if code not in {"GMAIL", "GITHUB", "JIRA", "MCP", "API"}
        ),
        None,
    )
    if project_code:
        queries = [project_code]
        expansion = _PROJECT_EXPANSIONS.get(project_code)
        if expansion:
            queries.append(f'"{expansion}"')
        return queries

    return [query]


def _wants_full_email(question: str) -> bool:
    return bool(re.search(r"\b(full|entire|complete|whole|body|read)\b", question, re.IGNORECASE))


def _wants_project_context(question: str) -> bool:
    return bool(
        re.search(
            r"\b(summarize|summary|progress|blockers?|owners?|updates?|status|project)\b",
            question,
            re.IGNORECASE,
        )
    )


async def _invoke(tool: Any, arguments: dict[str, Any]) -> Any:
    result = await tool.ainvoke(_validate_arguments(tool, arguments))
    normalized = _normalize_mcp_result(result)
    if isinstance(normalized, str) and normalized.lstrip().startswith("Error executing tool"):
        raise RuntimeError(normalized)
    if isinstance(normalized, dict) and normalized.get("isError"):
        raise RuntimeError(str(normalized.get("error", "Gmail MCP tool failed.")))
    return result


async def retrieve_gmail(subquestion: str) -> dict[str, Any]:
    question = subquestion.strip()
    if not question:
        raise ValueError("A Gmail question is required.")

    client = get_gmail_client()
    tools = await client.get_tools()
    tool_map = {tool.name: tool for tool in tools if tool.name in READ_ONLY_GMAIL_TOOLS}

    lower_question = question.lower()
    if any(word in lower_question for word in ("label", "labels", "folders", "mailbox folders")):
        tool_name = "list_labels"
        arguments: dict[str, Any] = {}
    else:
        tool_name = "search_emails"
        queries = _gmail_search_queries(question)
        arguments = {"query": queries[0], "max_results": 10}

    tool = tool_map.get(tool_name)
    if tool is None:
        raise RuntimeError(f"The read-only Gmail MCP tool {tool_name!r} is unavailable.")

    if tool_name == "search_emails":
        collected: dict[str, dict[str, Any]] = {}
        searched_queries = []
        for query in queries:
            searched_queries.append(query)
            raw_result = await _invoke(tool, {"query": query, "max_results": 10})
            query_evidence = _normalize_mcp_result(raw_result)
            if isinstance(query_evidence, dict):
                for message in query_evidence.get("messages", []):
                    if isinstance(message, dict) and message.get("id"):
                        collected[message["id"]] = message
        evidence = {
            "query": searched_queries,
            "count": len(collected),
            "messages": list(collected.values())[:10],
        }
    else:
        raw_result = await _invoke(tool, arguments)
        evidence = _normalize_mcp_result(raw_result)

    # Gmail search returns headers and snippets. For explicit full-message
    # requests, use the matched IDs with get_email to retrieve the body.
    include_full_email = _wants_full_email(question)
    if tool_name == "search_emails" and (include_full_email or _wants_project_context(question)):
        messages = evidence.get("messages", []) if isinstance(evidence, dict) else []
        get_email_tool = tool_map.get("get_email")
        if messages and get_email_tool is not None:
            full_emails = []
            maximum_bodies = 3 if include_full_email else 5
            for message in messages[:maximum_bodies]:
                message_id = message.get("id") if isinstance(message, dict) else None
                if message_id:
                    full_result = await _invoke(get_email_tool, {"message_id": message_id})
                    full_emails.append(_normalize_mcp_result(full_result))
            evidence = {**evidence, "full_emails": full_emails}

    return {
        "tool": "get_email" if tool_name == "search_emails" and include_full_email else tool.name,
        "authenticated": is_gmail_authenticated(),
        "evidence": evidence,
    }

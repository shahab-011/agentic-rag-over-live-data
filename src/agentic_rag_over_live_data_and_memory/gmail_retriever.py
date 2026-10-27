import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

READ_ONLY_GMAIL_TOOLS = frozenset({"search_emails", "get_email", "list_labels"})


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
    return token_path.is_file()


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
        arguments = {"query": question, "max_results": 10}

    tool = tool_map.get(tool_name)
    if tool is None:
        raise RuntimeError(f"The read-only Gmail MCP tool {tool_name!r} is unavailable.")

    validated = _validate_arguments(tool, arguments)
    raw_result = await tool.ainvoke(validated)
    return {
        "tool": tool.name,
        "authenticated": is_gmail_authenticated(),
        "evidence": _normalize_mcp_result(raw_result),
    }

import json

import pytest

from agentic_rag_over_live_data_and_memory import gmail_retriever


def test_gmail_configuration_uses_oauth_client_path(monkeypatch, tmp_path) -> None:
    client_path = tmp_path / "gmail-oauth-client.json"
    token_path = tmp_path / "gmail-token.json"
    client_path.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("GMAIL_OAUTH_CLIENT_PATH", str(client_path))
    monkeypatch.setenv("GMAIL_OAUTH_TOKEN_PATH", str(token_path))

    assert gmail_retriever.is_gmail_configured()
    assert not gmail_retriever.is_gmail_authenticated()


def test_gmail_authentication_reports_existing_token(monkeypatch, tmp_path) -> None:
    client_path = tmp_path / "gmail-oauth-client.json"
    token_path = tmp_path / "gmail-token.json"
    client_path.write_text("{}", encoding="utf-8")
    token_path.write_text(
        json.dumps({
            "token": "access-token",
            "refresh_token": "refresh-token",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "client-id",
            "client_secret": "client-secret",
            "scopes": [gmail_retriever.READ_ONLY_SCOPE],
        }),
        encoding="utf-8",
    )

    monkeypatch.setenv("GMAIL_OAUTH_CLIENT_PATH", str(client_path))
    monkeypatch.setenv("GMAIL_OAUTH_TOKEN_PATH", str(token_path))

    assert gmail_retriever.is_gmail_configured()
    assert gmail_retriever.is_gmail_authenticated()


def test_gmail_tool_allowlist_excludes_mutating_tools() -> None:
    assert gmail_retriever.READ_ONLY_GMAIL_TOOLS == {
        "search_emails",
        "get_email",
        "list_labels",
    }

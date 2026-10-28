import os
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from mcp.server.fastmcp import FastMCP

load_dotenv()

READ_ONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
server = FastMCP("gmail")


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


def get_gmail_service() -> Any:
    client_path, token_path = _oauth_paths()
    if not client_path.is_file():
        raise ValueError(f"Gmail OAuth client JSON not found: {client_path}")

    credentials: Credentials | None = None
    if token_path.is_file():
        # Read the scopes that Google actually granted from the token file.
        credentials = Credentials.from_authorized_user_file(str(token_path))

    if credentials is None or not credentials.valid:
        if credentials is not None and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(client_path),
                scopes=[READ_ONLY_SCOPE],
            )
            credentials = flow.run_local_server(port=0, open_browser=True)

        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(credentials.to_json(), encoding="utf-8")

    if READ_ONLY_SCOPE not in set(credentials.scopes or ()):
        raise ValueError("Existing Gmail token does not grant the read-only Gmail scope.")

    return build("gmail", "v1", credentials=credentials, cache_discovery=False)


def _plain_text(payload: dict[str, Any]) -> str:
    html_parts: list[str] = []

    def visit(part: dict[str, Any]) -> str | None:
        if part.get("mimeType") in {"text/plain", "text/html"} and isinstance(part.get("body"), dict):
            data = part["body"].get("data")
            if isinstance(data, str):
                import base64
                decoded = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode(
                    "utf-8", errors="replace"
                )
                if part.get("mimeType") == "text/plain":
                    return decoded
                html_parts.append(decoded)
        for child in part.get("parts", []):
            text = visit(child)
            if text:
                return text
        return None

    text = visit(payload.get("payload", {}))
    if text:
        return text
    if not html_parts:
        return ""

    class _TextExtractor(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.parts: list[str] = []

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag in {"br", "p", "div", "li", "tr", "h1", "h2", "h3"}:
                self.parts.append("\n")

        def handle_endtag(self, tag: str) -> None:
            if tag in {"p", "div", "li", "tr", "h1", "h2", "h3"}:
                self.parts.append("\n")

        def handle_data(self, data: str) -> None:
            self.parts.append(data)

    extractor = _TextExtractor()
    extractor.feed(html_parts[0])
    text = unescape("".join(extractor.parts))
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    paragraphs: list[str] = []
    for line in lines:
        if line:
            paragraphs.append(line)
        elif paragraphs and paragraphs[-1] != "":
            paragraphs.append("")
    return "\n".join(paragraphs).strip()


def _message_url(message_id: str) -> str:
    return f"https://mail.google.com/mail/u/0/#all/{message_id}"


@server.tool()
def search_emails(query: str, max_results: int = 10) -> str:
    """Search Gmail using Gmail search syntax without changing any messages."""
    if not query.strip():
        raise ValueError("A Gmail search query is required.")
    if not 1 <= max_results <= 50:
        raise ValueError("max_results must be between 1 and 50.")

    service = get_gmail_service()
    response = service.users().messages().list(
        userId="me",
        q=query.strip(),
        maxResults=max_results,
    ).execute()

    results = []
    for message in response.get("messages", [])[:max_results]:
        detail = service.users().messages().get(
            userId="me",
            id=message["id"],
            format="metadata",
            metadataHeaders=["Subject", "From", "To", "Date"],
        ).execute()
        headers = {
            item.get("name", ""): item.get("value", "")
            for item in detail.get("payload", {}).get("headers", [])
        }
        results.append({
            "id": message["id"],
            "url": _message_url(message["id"]),
            "thread_id": detail.get("threadId"),
            "subject": headers.get("Subject", ""),
            "from": headers.get("From", ""),
            "to": headers.get("To", ""),
            "date": headers.get("Date", ""),
            "snippet": detail.get("snippet", ""),
        })

    import json
    return json.dumps({"query": query.strip(), "count": len(results), "messages": results}, ensure_ascii=False)


@server.tool()
def get_email(message_id: str) -> str:
    """Retrieve read-only content and headers for one Gmail message."""
    if not message_id.strip():
        raise ValueError("A Gmail message ID is required.")

    service = get_gmail_service()
    message = service.users().messages().get(
        userId="me",
        id=message_id.strip(),
        format="full",
    ).execute()
    headers = {
        item.get("name", ""): item.get("value", "")
        for item in message.get("payload", {}).get("headers", [])
    }
    import json
    return json.dumps({
        "id": message.get("id"),
        "url": _message_url(message.get("id", "")),
        "thread_id": message.get("threadId"),
        "subject": headers.get("Subject", ""),
        "from": headers.get("From", ""),
        "to": headers.get("To", ""),
        "date": headers.get("Date", ""),
        "snippet": message.get("snippet", ""),
        "body": _plain_text(message),
    }, ensure_ascii=False)


@server.tool()
def list_labels() -> str:
    """List Gmail labels without changing mailbox state."""
    service = get_gmail_service()
    response = service.users().labels().list(userId="me").execute()
    labels = [{
        "id": label.get("id"),
        "name": label.get("name"),
        "type": label.get("type"),
    } for label in response.get("labels", [])]
    import json
    return json.dumps({"count": len(labels), "labels": labels}, ensure_ascii=False)


if __name__ == "__main__":
    server.run(transport="stdio")

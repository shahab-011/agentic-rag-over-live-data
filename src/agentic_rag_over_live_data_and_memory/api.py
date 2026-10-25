import asyncio
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from dotenv import load_dotenv

from agentic_rag_over_live_data_and_memory.jira_retriever import (
    is_jira_configured,
)
from agentic_rag_over_live_data_and_memory.workflow import run_workflow

load_dotenv()

LOGGER = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 64 * 1024


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "AgenticRagApi/1.0"

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        response = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.send_header("Access-Control-Allow-Origin", "http://localhost:5173")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(response)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "http://localhost:5173")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self._send_json(200, {"status": "ok"})
            return
        if self.path == "/api/sources":
            self._send_json(
                200,
                {
                    "github": bool(os.getenv("GITHUB_TOKEN")),
                    "notion": bool(os.getenv("NOTION_API_KEY")),
                    "jira": is_jira_configured(),
                },
            )
            return
        self._send_json(404, {"error": "Not found."})

    def do_POST(self) -> None:
        if self.path != "/api/ask":
            self._send_json(404, {"error": "Not found."})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length header."})
            return

        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            self._send_json(
                413 if content_length > MAX_REQUEST_BYTES else 400,
                {"error": "Request body must be between 1 byte and 64 KB."},
            )
            return

        try:
            payload = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"error": "Request body must be valid JSON."})
            return

        question = payload.get("question") if isinstance(payload, dict) else None
        if not isinstance(question, str) or not question.strip():
            self._send_json(400, {"error": "Enter a question to continue."})
            return

        try:
            result = asyncio.run(run_workflow(question))
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except Exception:
            LOGGER.exception("Workflow request failed")
            self._send_json(
                500,
                {
                    "error": (
                        "The assistant could not complete this request. "
                        "Check the API server logs for details."
                    )
                },
            )
            return

        self._send_json(200, result.model_dump(mode="json"))

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), format % args)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = ThreadingHTTPServer(("127.0.0.1", 8000), ApiHandler)
    LOGGER.info("Agentic RAG API listening at http://127.0.0.1:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down API server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

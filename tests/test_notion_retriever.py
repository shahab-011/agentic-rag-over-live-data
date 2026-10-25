import asyncio

import httpx
import pytest

from agentic_rag_over_live_data_and_memory import notion_retriever


def test_retrieve_notion_searches_shared_pages_and_reads_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOTION_API_KEY", "test-token")
    requests: list[httpx.Request] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer test-token"
        assert request.headers["Notion-Version"] == "2022-06-28"
        if request.method == "POST":
            assert request.url.path == "/v1/search"
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "object": "page",
                            "id": "page-id",
                            "url": "https://www.notion.so/example-page",
                            "properties": {
                                "Name": {
                                    "type": "title",
                                    "title": [{"plain_text": "Roadmap"}],
                                }
                            },
                        }
                    ]
                },
            )
        assert request.url.path == "/v1/blocks/page-id/children"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "type": "heading_1",
                        "heading_1": {
                            "rich_text": [{"plain_text": "Launch plan"}]
                        },
                    },
                    {
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"plain_text": "Ship in November."}]
                        },
                    },
                ],
                "has_more": False,
                "next_cursor": None,
            },
        )

    transport = httpx.MockTransport(handle_request)
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        notion_retriever.httpx,
        "AsyncClient",
        lambda **kwargs: real_async_client(transport=transport, **kwargs),
    )

    result = asyncio.run(notion_retriever.retrieve_notion("project launch plan"))

    assert result["tool"] == "notion_search"
    assert result["evidence"]["pages"] == [
        {
            "title": "Roadmap",
            "url": "https://www.notion.so/example-page",
            "content": ["Launch plan", "Ship in November."],
        }
    ]
    assert len(requests) == 2


def test_retrieve_notion_requires_an_integration_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NOTION_API_KEY", raising=False)

    with pytest.raises(ValueError, match="NOTION_API_KEY"):
        asyncio.run(notion_retriever.retrieve_notion("Find a page"))

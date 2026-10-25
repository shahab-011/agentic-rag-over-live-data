import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_API_VERSION = "2022-06-28"
MAX_SEARCH_RESULTS = 5
MAX_BLOCKS_PER_PAGE = 100
MAX_PAGE_CONTENT_CHARS = 12_000
MAX_EVIDENCE_CHUNK_CHARS = 300
MAX_NESTED_BLOCK_DEPTH = 2


def is_notion_configured() -> bool:
    return bool(os.getenv("NOTION_API_KEY"))


def _plain_text(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    fragments: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = item.get("plain_text")
        if not isinstance(text, str):
            rich_text = item.get("text")
            text = (
                rich_text.get("content", "")
                if isinstance(rich_text, dict)
                else ""
            )
        if isinstance(text, str):
            fragments.append(text)
    return "".join(fragments)


def _page_title(page: dict[str, Any]) -> str:
    properties = page.get("properties", {})
    if isinstance(properties, dict):
        for prop in properties.values():
            if isinstance(prop, dict) and prop.get("type") == "title":
                title = _plain_text(prop.get("title"))
                if title:
                    return title
    title = _plain_text(page.get("title"))
    return title or "Untitled"


async def _block_text(
    client: httpx.AsyncClient,
    block_id: str,
    *,
    depth: int = 0,
    remaining_blocks: list[int],
) -> list[str]:
    lines: list[str] = []
    cursor: str | None = None

    while remaining_blocks[0] > 0:
        params: dict[str, Any] = {"page_size": min(100, remaining_blocks[0])}
        if cursor:
            params["start_cursor"] = cursor
        response = await client.get(
            f"{NOTION_API_BASE}/blocks/{block_id}/children",
            params=params,
        )
        response.raise_for_status()
        payload = response.json()
        blocks = payload.get("results")
        if not isinstance(blocks, list):
            raise TypeError("Notion returned an invalid block list.")

        for block in blocks:
            if remaining_blocks[0] <= 0:
                break
            remaining_blocks[0] -= 1
            if not isinstance(block, dict):
                continue

            block_type = block.get("type")
            content = block.get(block_type, {})
            if isinstance(content, dict):
                text = _plain_text(content.get("rich_text"))
                if not text:
                    text = _plain_text(content.get("title"))
                if text:
                    lines.append(text)

            child_id = block.get("id")
            if (
                block.get("has_children")
                and isinstance(child_id, str)
                and depth < MAX_NESTED_BLOCK_DEPTH
                and remaining_blocks[0] > 0
            ):
                lines.extend(
                    await _block_text(
                        client,
                        child_id,
                        depth=depth + 1,
                        remaining_blocks=remaining_blocks,
                    )
                )

        has_more = payload.get("has_more")
        if not isinstance(has_more, bool):
            raise TypeError("Notion returned an invalid pagination response.")
        if not has_more or remaining_blocks[0] <= 0:
            break
        cursor = payload.get("next_cursor")
        if not isinstance(cursor, str) or not cursor:
            raise RuntimeError("Notion omitted the next cursor for more blocks.")

    return lines


async def retrieve_notion(subquestion: str) -> dict[str, Any]:
    """Search pages shared with the integration and retrieve their content."""

    token = os.getenv("NOTION_API_KEY")
    if not token:
        raise ValueError("NOTION_API_KEY is missing from .env.")

    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_API_VERSION,
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(headers=headers, timeout=30) as client:
        response = await client.post(
            f"{NOTION_API_BASE}/search",
            json={
                "query": subquestion[:100],
                "page_size": MAX_SEARCH_RESULTS,
                "filter": {"property": "object", "value": "page"},
            },
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results")
        if not isinstance(results, list):
            raise TypeError("Notion returned an invalid search response.")

        pages: list[dict[str, Any]] = []
        for item in results[:MAX_SEARCH_RESULTS]:
            if not isinstance(item, dict) or item.get("object") != "page":
                continue
            page_id = item.get("id")
            page_url = item.get("url")
            if not isinstance(page_id, str) or not isinstance(page_url, str):
                continue
            remaining_blocks = [MAX_BLOCKS_PER_PAGE]
            lines = await _block_text(
                client,
                page_id,
                remaining_blocks=remaining_blocks,
            )
            content: list[str] = []
            remaining_chars = MAX_PAGE_CONTENT_CHARS
            for line in lines:
                bounded_line = line[:remaining_chars]
                content.extend(
                    bounded_line[index:index + MAX_EVIDENCE_CHUNK_CHARS]
                    for index in range(
                        0,
                        len(bounded_line),
                        MAX_EVIDENCE_CHUNK_CHARS,
                    )
                )
                remaining_chars -= len(bounded_line)
                if remaining_chars == 0:
                    break
            pages.append(
                {
                    "title": _page_title(item),
                    "url": page_url,
                    "content": content,
                }
            )

    return {
        "tool": "notion_search",
        "evidence": {"pages": pages},
    }

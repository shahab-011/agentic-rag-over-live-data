import asyncio
import os

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

async def main():
    token = os.getenv("GITHUB_TOKEN")

    if not token:
        raise ValueError("GITHUB_TOKEN is missing from .env")

    client = MultiServerMCPClient(
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

    tools = await client.get_tools()
    
   
    for tool in tools:
        if tool.name in {
            "list_commits",
            "list_issues",
            "list_pull_requests",
            "search_repositories",
        }:
            print(f"\nTool: {tool.name}")
            print("Input schema:")
            print(tool.args)


    # search_tool = next(
    #     (tool for tool in tools if tool.name == "search_repositories"),
    #     None,
    # )

    # if search_tool is None:
    #     raise RuntimeError("search_repositories tool was not found")

    # result = await search_tool.ainvoke(
    #     {"query": "langgraph"}
    # )

    # print("\nGitHub repository search result:\n")
    # print(str(result)[:5000])


if __name__ == "__main__":
    asyncio.run(main())


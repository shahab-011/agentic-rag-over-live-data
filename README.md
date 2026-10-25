# Agentic RAG over live data and memory

The LangGraph workflow decomposes a question, routes its subquestions to
source-specific retrieval nodes, retrieves evidence, and uses Groq to generate
a grounded final answer. GitHub requests use the GitHub MCP retriever, and
Notion requests search pages shared with your Notion integration. Gmail has a
placeholder node until that integration is implemented.
Factual claims in the answer use numbered citations linked to URLs found in
the retrieved evidence. Citations that do not match retrieved sources are
rejected.

For GitHub retrieval, Groq selects a read-only MCP tool and prepares its
arguments from the subquestion and the live tool schemas. The backend validates
the selected tool, required arguments, types, and enum values before invoking
it. Personal repository-count questions use the authenticated GitHub profile
API instead.

File-content requests fetch the complete text file from the GitHub Contents API
and render it verbatim with a link to the source, rather than summarizing or
truncating the code through Groq.

Configure `GROQ_API_KEY`, `GROQ_MODEL`, and the GitHub settings in `.env`.
Set `GITHUB_OWNER` and `GITHUB_REPO` to the repository Orbit should use when a
question does not name one; the defaults are `langchain-ai` and `langgraph`.

To connect Jira Cloud:

1. Add your Jira site URL, Atlassian account email, and Jira API token to
   `.env` as `JIRA_URL`, `JIRA_USERNAME`, and `JIRA_API_KEY`.
2. Use the site's base URL for `JIRA_URL` (for example,
   `https://your-company.atlassian.net`), not a project or board URL.
3. Restart the API. The local, read-only Jira MCP server is started on demand
   by the backend and uses the configured account to search issues, read an
   issue, and list projects.

Jira retrieval is read-only. The MCP server exposes only issue search, issue
lookup, and project listing, and authenticates to Jira Cloud using the account
email and API token. Keep `.env` private and never commit the token.
After routing a question to Jira, Orbit uses a Jira-specific AI refinement
step to turn the subquestion into a Jira-friendly request and, for issue
searches, JQL. It then validates the operation and arguments before calling
the read-only MCP tool. The research trail shows the refined Jira question.

To connect Notion:

1. Create an internal integration in your Notion workspace and copy its secret.
2. Add `NOTION_API_KEY=<your integration secret>` to `.env`.
3. In Notion, share each page or parent page you want Orbit to search with
   that integration. The integration can only search pages shared with it.
4. Restart the API. The Connected sources panel reports whether the token is
   configured.

Orbit uses Notion's read-only search and page-block endpoints. It searches up
to five matching pages per subquestion and retrieves a bounded amount of page
content, with links back to the source pages in citations.
Start the API from the project root in PowerShell:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m agentic_rag_over_live_data_and_memory.api
```

In a second PowerShell window, install and run the React interface:

```powershell
Set-Location frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal (normally http://127.0.0.1:5173).
The frontend proxies API requests to the local backend on port 8000.
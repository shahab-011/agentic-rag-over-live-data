
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, HttpUrl


class GitHubUser(BaseModel):
    login: str | None = None
    html_url: HttpUrl | None = None


class PullRequest(BaseModel):
    number: int
    title: str
    state: str | None = None
    html_url: HttpUrl
    created_at: str | None = None
    updated_at: str | None = None
    user: GitHubUser | None = None


class GitHubIssue(BaseModel):
    number: int
    title: str
    state: str | None = None
    html_url: HttpUrl
    created_at: str | None = None
    updated_at: str | None = None
    user: GitHubUser | None = None


class GitHubCommit(BaseModel):
    sha: str
    html_url: HttpUrl | None = None
    message: str | None = None


class GitHubRepository(BaseModel):
    name: str
    full_name: str | None = None
    html_url: HttpUrl
    description: str | None = None


class GitHubEvidence(BaseModel):
    tool: str
    items: list[dict[str, Any]] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    subquestion: str
    source: Literal["github", "gmail", "notion", "jira"]
    status: Literal["success", "error", "not_implemented"]
    result: dict[str, Any] | None = None
    error: str | None = None


class AnswerDraft(BaseModel):
    answer: str = Field(
        validation_alias=AliasChoices("answer", "summary", "final_answer"),
    )


class Citation(BaseModel):
    number: int
    source: str
    subquestion: str
    url: str


class WorkflowOutput(BaseModel):
    original_question: str
    subquestions: list[str] = Field(default_factory=list)
    routes: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_results: list[RetrievalResult] = Field(
        default_factory=list
    )
    final_answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)

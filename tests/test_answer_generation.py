import pytest

from agentic_rag_over_live_data_and_memory import answer_generation
from agentic_rag_over_live_data_and_memory.schemas import (
    AnswerDraft,
    RetrievalResult,
)


class FakeStructuredLlm:
    def __init__(self, draft: AnswerDraft) -> None:
        self.draft = draft

    def invoke(self, messages: list[tuple[str, str]]) -> AnswerDraft:
        return self.draft


class FakeChatGroq:
    def __init__(
        self,
        draft: AnswerDraft,
        **kwargs: object,
    ) -> None:
        self.draft = draft

    def with_structured_output(
        self,
        schema: type[AnswerDraft],
        **kwargs: object,
    ) -> FakeStructuredLlm:
        return FakeStructuredLlm(self.draft)


def test_generate_cited_answer_returns_only_retrieved_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "test-model")
    monkeypatch.setattr(
        answer_generation,
        "ChatGroq",
        lambda **kwargs: FakeChatGroq(
            AnswerDraft(
                answer="A recent commit is available [1].",
            ),
            **kwargs,
        ),
    )
    results = [
        RetrievalResult(
            subquestion="List recent commits.",
            source="github",
            status="success",
            result={
                "evidence": [
                    {"html_url": "https://github.com/org/repo/commit/abc"}
                ]
            },
        ),
        RetrievalResult(
            subquestion="Other source failed.",
            source="gmail",
            status="error",
            error="Not connected.",
        ),
    ]

    answer, citations = answer_generation.generate_cited_answer(
        "What changed recently?",
        results,
    )

    assert answer == "A recent commit is available [1]."
    assert [citation.url for citation in citations] == [
        "https://github.com/org/repo/commit/abc"
    ]


def test_generate_cited_answer_rejects_unknown_citation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "test-model")
    monkeypatch.setattr(
        answer_generation,
        "ChatGroq",
        lambda **kwargs: FakeChatGroq(
            AnswerDraft(answer="Unsupported claim [99]."),
            **kwargs,
        ),
    )
    results = [
        RetrievalResult(
            subquestion="List recent commits.",
            source="github",
            status="success",
            result={"evidence": [{"html_url": "https://github.com/org/repo"}]},
        )
    ]

    with pytest.raises(ValueError, match="not present"):
        answer_generation.generate_cited_answer("Question?", results)


def test_generate_cited_answer_rejects_unretrieved_citation_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "test-model")
    monkeypatch.setattr(
        answer_generation,
        "ChatGroq",
        lambda **kwargs: FakeChatGroq(
            AnswerDraft(
                answer="A claim [1] with https://example.invalid/source."
            ),
            **kwargs,
        ),
    )
    results = [
        RetrievalResult(
            subquestion="List recent commits.",
            source="github",
            status="success",
            result={"evidence": [{"html_url": "https://github.com/org/repo"}]},
        )
    ]

    with pytest.raises(ValueError, match="not present"):
        answer_generation.generate_cited_answer("Question?", results)


def test_repository_list_answer_includes_all_repositories_without_groq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repositories = [
        {
            "name": f"project-{index}",
            "full_name": f"test-user/project-{index}",
            "html_url": f"https://github.com/test-user/project-{index}",
            "private": index == 7,
        }
        for index in range(1, 9)
    ]
    result = RetrievalResult(
        subquestion="List all my repositories.",
        source="github",
        status="success",
        result={
            "tool": "get_authenticated_repository_list",
            "evidence": {
                "login": "test-user",
                "total_repositories": len(repositories),
                "public_repositories": 7,
                "private_repositories": 1,
                "repositories": repositories,
            },
        },
    )
    monkeypatch.setattr(
        answer_generation,
        "ChatGroq",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("Complete repository lists should not be LLM-truncated")
        ),
    )

    answer, citations = answer_generation.generate_cited_answer(
        "Generate all repository names.",
        [result],
    )

    for repository in repositories:
        assert f"`{repository['full_name']}`" in answer
    assert "(private)" in answer
    assert len(citations) == len(repositories)
    assert "[8]" in answer


def test_file_contents_answer_includes_complete_source_without_groq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_content = "def example():\n    return '" + ("x" * 500) + "'\n"
    result = RetrievalResult(
        subquestion="Show the complete code in src/example.py.",
        source="github",
        status="success",
        result={
            "tool": "get_file_contents",
            "evidence": {
                "path": "src/example.py",
                "file_content": file_content,
                "html_url": (
                    "https://github.com/test-user/project/blob/main/"
                    "src/example.py"
                ),
            },
        },
    )
    monkeypatch.setattr(
        answer_generation,
        "ChatGroq",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("Exact file contents should not be sent to Groq")
        ),
    )

    answer, citations = answer_generation.generate_cited_answer(
        "Show the complete code in src/example.py.",
        [result],
    )

    assert file_content in answer
    assert answer.startswith("Complete contents of `src/example.py` [1]:")
    assert len(citations) == 1
    assert citations[0].url == (
        "https://github.com/test-user/project/blob/main/src/example.py"
    )

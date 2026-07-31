from dataclasses import dataclass
from datetime import date
from typing import Any

from langchain_core.documents import Document
from langchain_ollama.llms import OllamaLLM

from ollama_health import DEFAULT_CHAT_MODEL, DEFAULT_OLLAMA_HOST
from vector import ReviewMatch, search_reviews

NO_MATCH_MESSAGE = (
    "I could not find any reviews matching the current question and filters."
)

ANSWER_PROMPT = """You are a restaurant review analyst.
Answer the question using only the supplied reviews. Do not add facts that are not present.
When evidence is mixed or limited, say so clearly. Cite supporting reviews with bracketed
numbers such as [1] or [2].

Question:
{question}

Supplied reviews:
{context}

Answer:
"""


@dataclass(frozen=True)
class CitedReview:
    citation_number: int
    document: Document
    score: float


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: tuple[CitedReview, ...]


def create_chat_model(
    *,
    model: str = DEFAULT_CHAT_MODEL,
    base_url: str = DEFAULT_OLLAMA_HOST,
) -> OllamaLLM:
    return OllamaLLM(model=model, base_url=base_url, temperature=0)


def _format_context(matches: list[ReviewMatch]) -> str:
    sections: list[str] = []
    for number, match in enumerate(matches, start=1):
        metadata = match.document.metadata
        sections.append(
            "\n".join(
                (
                    f"[{number}] Rating: {metadata.get('rating', 'unknown')}/5",
                    f"Date: {metadata.get('date', 'unknown')}",
                    match.document.page_content,
                )
            )
        )
    return "\n\n".join(sections)


def answer_question(
    question: str,
    *,
    vector_store: Any,
    model: Any | None = None,
    limit: int = 5,
    min_rating: int = 1,
    max_rating: int = 5,
    start_date: date | None = None,
    end_date: date | None = None,
    chat_model: str = DEFAULT_CHAT_MODEL,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
) -> AnswerResult:
    """Retrieve filtered evidence and produce a cited, grounded answer."""
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("question cannot be empty")

    matches = search_reviews(
        vector_store,
        normalized_question,
        limit=limit,
        min_rating=min_rating,
        max_rating=max_rating,
        start_date=start_date,
        end_date=end_date,
    )
    if not matches:
        return AnswerResult(answer=NO_MATCH_MESSAGE, sources=())

    answer_model = model or create_chat_model(model=chat_model, base_url=ollama_host)
    prompt = ANSWER_PROMPT.format(
        question=normalized_question,
        context=_format_context(matches),
    )
    response = answer_model.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    sources = tuple(
        CitedReview(
            citation_number=number,
            document=match.document,
            score=match.score,
        )
        for number, match in enumerate(matches, start=1)
    )
    return AnswerResult(answer=answer.strip(), sources=sources)

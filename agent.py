import re
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
CITATION_VALIDATION_MESSAGE = (
    "I could not produce an answer with citations that match the retrieved reviews."
)
CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9_-]*)\]")
INSUFFICIENT_EVIDENCE_TOKEN = "INSUFFICIENT_EVIDENCE"

ANSWER_PROMPT = """You are a review analyst.
Answer the question using only the supplied reviews. Do not add facts that are not present.
When evidence is mixed or limited, say so clearly. Every factual claim must cite one or more
retrieved evidence numbers exactly as shown, for example [1]. Never cite an evidence number
that is not supplied. Source IDs are validation metadata; do not copy them into the answer.
If the supplied reviews do not answer the question, reply exactly INSUFFICIENT_EVIDENCE.

Question:
{question}

Supplied review records:
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
    retrieved_source_ids: tuple[str, ...] = ()
    abstained: bool = False


def create_chat_model(
    *,
    model: str = DEFAULT_CHAT_MODEL,
    base_url: str = DEFAULT_OLLAMA_HOST,
) -> OllamaLLM:
    return OllamaLLM(model=model, base_url=base_url, temperature=0)


def _format_context(matches: list[ReviewMatch]) -> str:
    sections: list[str] = []
    labels = (
        ("rating", "Rating"),
        ("date", "Date"),
        ("sentiment", "Sentiment"),
        ("restaurant", "Restaurant"),
        ("country", "Country"),
    )
    for evidence_number, match in enumerate(matches, start=1):
        metadata = match.document.metadata
        source_id = str(metadata.get("source_id") or match.document.id or "")
        if not source_id:
            raise ValueError("retrieved review is missing a source ID")
        lines = [f"[{evidence_number}]", f"Source ID: {source_id}"]
        for key, label in labels:
            value = metadata.get(key)
            if value is not None:
                suffix = "/5" if key == "rating" else ""
                lines.append(f"{label}: {value}{suffix}")
        lines.append(match.document.page_content)
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _validate_and_number_citations(
    answer: str,
    matches: list[ReviewMatch],
) -> tuple[str, tuple[CitedReview, ...]] | None:
    retrieved: dict[str, ReviewMatch] = {}
    evidence_aliases: dict[str, str] = {}
    for evidence_number, match in enumerate(matches, start=1):
        source_id = str(
            match.document.metadata.get("source_id") or match.document.id or ""
        )
        if not source_id:
            return None
        retrieved[source_id] = match
        evidence_aliases[str(evidence_number)] = source_id

    cited_tokens = CITATION_PATTERN.findall(answer)
    if not cited_tokens:
        return None

    resolved_ids: list[str] = []
    for token in cited_tokens:
        source_id = evidence_aliases.get(token, token)
        if source_id not in retrieved:
            return None
        resolved_ids.append(source_id)

    ordered_ids = list(dict.fromkeys(resolved_ids))
    citation_numbers = {
        source_id: number for number, source_id in enumerate(ordered_ids, start=1)
    }
    numbered_answer = CITATION_PATTERN.sub(
        lambda match: (
            f"[{citation_numbers[evidence_aliases.get(match.group(1), match.group(1))]}]"
        ),
        answer,
    )
    sources = tuple(
        CitedReview(
            citation_number=citation_numbers[source_id],
            document=retrieved[source_id].document,
            score=retrieved[source_id].score,
        )
        for source_id in ordered_ids
    )
    return numbered_answer, sources


def answer_question(
    question: str,
    *,
    vector_store: Any,
    model: Any | None = None,
    limit: int = 5,
    min_rating: int | None = None,
    max_rating: int | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    sentiments: tuple[str, ...] | list[str] = (),
    restaurants: tuple[str, ...] | list[str] = (),
    countries: tuple[str, ...] | list[str] = (),
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
        sentiments=sentiments,
        restaurants=restaurants,
        countries=countries,
    )
    if not matches:
        return AnswerResult(answer=NO_MATCH_MESSAGE, sources=())

    retrieved_source_ids = tuple(
        str(match.document.metadata.get("source_id") or match.document.id or "")
        for match in matches
    )
    if any(not source_id for source_id in retrieved_source_ids):
        return AnswerResult(answer=CITATION_VALIDATION_MESSAGE, sources=())

    answer_model = model or create_chat_model(model=chat_model, base_url=ollama_host)
    prompt = ANSWER_PROMPT.format(
        question=normalized_question,
        context=_format_context(matches),
    )
    response = answer_model.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    normalized_answer = answer.strip()
    if normalized_answer == INSUFFICIENT_EVIDENCE_TOKEN:
        return AnswerResult(
            answer=NO_MATCH_MESSAGE,
            sources=(),
            retrieved_source_ids=retrieved_source_ids,
            abstained=True,
        )
    validated = _validate_and_number_citations(normalized_answer, matches)
    if validated is None:
        return AnswerResult(
            answer=CITATION_VALIDATION_MESSAGE,
            sources=(),
            retrieved_source_ids=retrieved_source_ids,
        )
    validated_answer, sources = validated
    return AnswerResult(
        answer=validated_answer,
        sources=sources,
        retrieved_source_ids=retrieved_source_ids,
    )

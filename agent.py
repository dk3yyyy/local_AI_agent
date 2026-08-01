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
REPAIRABLE_FAILURE_REASONS = frozenset(
    {"missing_citations", "out_of_range_citation", "unknown_citation"}
)

ANSWER_PROMPT = """You are a review analyst.
Answer the question using only the supplied reviews. Do not add facts that are not present.
Always answer every part that at least one supplied review supports. Mixed or incomplete evidence is
not a reason to abstain; describe the limitation and answer only the supported portion. Every
factual claim must cite one or more retrieved evidence numbers exactly as shown, for example
[1]. Never cite an evidence number that is not supplied. Source IDs are validation metadata;
do not copy them into the answer. Reply exactly INSUFFICIENT_EVIDENCE only when no supplied
review answers any part of the question. Never use INSUFFICIENT_EVIDENCE as prose or place it
beside an answer.

Question:
{question}

Supplied review records:
{context}

Answer:
"""

REPAIR_PROMPT = """You are repairing one rejected review-analysis response.
Using only the supplied reviews, rewrite it once so every factual claim has one or more valid
evidence citations such as [1]. Use only evidence numbers that appear below. Preserve supported
meaning, remove unsupported claims, and answer every supported part even when evidence is mixed
or incomplete. If no supplied review answers any part, reply exactly INSUFFICIENT_EVIDENCE.
Never use INSUFFICIENT_EVIDENCE as prose or place it beside an answer.

Question:
{question}

Supplied review records:
{context}

Rejected response:
<rejected_response>
{rejected_response}
</rejected_response>

Rewritten answer:
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
    raw_response: str = ""
    repair_response: str | None = None
    initial_failure_reason: str | None = None
    failure_reason: str | None = None
    repair_attempted: bool = False


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


def _remove_standalone_control_token(answer: str) -> str | None:
    if INSUFFICIENT_EVIDENCE_TOKEN not in answer:
        return answer
    retained_lines: list[str] = []
    for line in answer.splitlines():
        if line.strip() == INSUFFICIENT_EVIDENCE_TOKEN:
            continue
        if INSUFFICIENT_EVIDENCE_TOKEN in line:
            return None
        retained_lines.append(line)
    return "\n".join(retained_lines).strip()


def _validate_and_number_citations(
    answer: str,
    matches: list[ReviewMatch],
) -> tuple[tuple[str, tuple[CitedReview, ...]] | None, str | None]:
    retrieved: dict[str, ReviewMatch] = {}
    evidence_aliases: dict[str, str] = {}
    for evidence_number, match in enumerate(matches, start=1):
        source_id = str(
            match.document.metadata.get("source_id") or match.document.id or ""
        )
        if not source_id:
            return None, "retrieved_source_missing_id"
        retrieved[source_id] = match
        evidence_aliases[str(evidence_number)] = source_id

    cited_tokens = CITATION_PATTERN.findall(answer)
    if not cited_tokens:
        return None, "missing_citations"

    resolved_ids: list[str] = []
    for token in cited_tokens:
        source_id = evidence_aliases.get(token, token)
        if source_id not in retrieved:
            reason = "out_of_range_citation" if token.isdigit() else "unknown_citation"
            return None, reason
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
    return (numbered_answer, sources), None


def _evaluate_model_response(
    response: str,
    matches: list[ReviewMatch],
) -> tuple[tuple[str, tuple[CitedReview, ...]] | None, str | None]:
    normalized = response.strip()
    if normalized == INSUFFICIENT_EVIDENCE_TOKEN:
        return None, "clean_abstention"
    normalized_without_control = _remove_standalone_control_token(normalized)
    if normalized_without_control is None:
        return None, "embedded_control_token"
    if not normalized_without_control:
        return None, "invalid_remainder"
    return _validate_and_number_citations(normalized_without_control, matches)


def _response_text(response: Any) -> str:
    return str(response.content if hasattr(response, "content") else response)


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
        return AnswerResult(
            answer=NO_MATCH_MESSAGE,
            sources=(),
            failure_reason="empty_retrieval",
        )

    retrieved_source_ids = tuple(
        str(match.document.metadata.get("source_id") or match.document.id or "")
        for match in matches
    )
    if any(not source_id for source_id in retrieved_source_ids):
        return AnswerResult(
            answer=CITATION_VALIDATION_MESSAGE,
            sources=(),
            failure_reason="retrieved_source_missing_id",
        )

    answer_model = model or create_chat_model(model=chat_model, base_url=ollama_host)
    context = _format_context(matches)
    prompt = ANSWER_PROMPT.format(
        question=normalized_question,
        context=context,
    )
    raw_response = _response_text(answer_model.invoke(prompt))
    validated, failure_reason = _evaluate_model_response(raw_response, matches)
    if validated is not None:
        validated_answer, sources = validated
        return AnswerResult(
            answer=validated_answer,
            sources=sources,
            retrieved_source_ids=retrieved_source_ids,
            raw_response=raw_response,
        )
    if failure_reason == "clean_abstention":
        return AnswerResult(
            answer=NO_MATCH_MESSAGE,
            sources=(),
            retrieved_source_ids=retrieved_source_ids,
            abstained=True,
            raw_response=raw_response,
            failure_reason=failure_reason,
        )
    if failure_reason not in REPAIRABLE_FAILURE_REASONS:
        return AnswerResult(
            answer=CITATION_VALIDATION_MESSAGE,
            sources=(),
            retrieved_source_ids=retrieved_source_ids,
            raw_response=raw_response,
            failure_reason=failure_reason,
        )

    initial_failure_reason = failure_reason
    repair_prompt = REPAIR_PROMPT.format(
        question=normalized_question,
        context=context,
        rejected_response=raw_response,
    )
    repair_response = _response_text(answer_model.invoke(repair_prompt))
    repaired, repair_failure_reason = _evaluate_model_response(repair_response, matches)
    if repaired is not None:
        repaired_answer, sources = repaired
        return AnswerResult(
            answer=repaired_answer,
            sources=sources,
            retrieved_source_ids=retrieved_source_ids,
            raw_response=raw_response,
            repair_response=repair_response,
            initial_failure_reason=initial_failure_reason,
            repair_attempted=True,
        )
    if repair_failure_reason == "clean_abstention":
        return AnswerResult(
            answer=NO_MATCH_MESSAGE,
            sources=(),
            retrieved_source_ids=retrieved_source_ids,
            abstained=True,
            raw_response=raw_response,
            repair_response=repair_response,
            initial_failure_reason=initial_failure_reason,
            failure_reason=repair_failure_reason,
            repair_attempted=True,
        )
    return AnswerResult(
        answer=CITATION_VALIDATION_MESSAGE,
        sources=(),
        retrieved_source_ids=retrieved_source_ids,
        raw_response=raw_response,
        repair_response=repair_response,
        initial_failure_reason=initial_failure_reason,
        failure_reason=repair_failure_reason,
        repair_attempted=True,
    )

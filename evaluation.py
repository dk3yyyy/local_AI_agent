import json
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from agent import NO_MATCH_MESSAGE, answer_question
from vector import search_reviews

DEFAULT_EVALUATION_PATH = Path(
    str(files("local_ai_agent.data").joinpath("rag_cases.json"))
)


@dataclass(frozen=True)
class ReferenceFact:
    answer_terms: tuple[str, ...]
    source_terms: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    relevant_titles: tuple[str, ...]
    reference_facts: tuple[ReferenceFact, ...]
    should_abstain: bool = False


@dataclass(frozen=True)
class EvaluationObservation:
    case_id: str
    relevant_source_ids: frozenset[str]
    retrieved_source_ids: tuple[str, ...]
    cited_source_ids: tuple[str, ...]
    cited_text: str
    answer: str
    abstained: bool


@dataclass(frozen=True)
class EvaluationMetrics:
    retrieval_recall: float
    citation_correctness: float
    answer_faithfulness: float
    abstention_accuracy: float
    case_count: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "retrieval_recall": self.retrieval_recall,
            "citation_correctness": self.citation_correctness,
            "answer_faithfulness": self.answer_faithfulness,
            "abstention_accuracy": self.abstention_accuracy,
            "case_count": self.case_count,
        }


def load_evaluation_cases(path: str | Path) -> tuple[EvaluationCase, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(
        EvaluationCase(
            case_id=item["id"],
            question=item["question"],
            relevant_titles=tuple(item.get("relevant_titles", ())),
            reference_facts=tuple(
                ReferenceFact(
                    answer_terms=tuple(fact["answer_terms"]),
                    source_terms=tuple(fact["source_terms"]),
                )
                for fact in item.get("reference_facts", ())
            ),
            should_abstain=bool(item.get("should_abstain", False)),
        )
        for item in payload
    )


def _source_id(document: Any) -> str:
    return str(document.metadata.get("source_id") or document.id or "")


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 1.0


def _faithfulness(case: EvaluationCase, observation: EvaluationObservation) -> float:
    if case.should_abstain:
        return (
            1.0 if observation.abstained and not observation.cited_source_ids else 0.0
        )
    if not case.reference_facts:
        return 1.0
    answer = observation.answer.casefold()
    sources = observation.cited_text.casefold()
    return _mean(
        1.0
        if any(term.casefold() in answer for term in fact.answer_terms)
        and any(term.casefold() in sources for term in fact.source_terms)
        else 0.0
        for fact in case.reference_facts
    )


def score_evaluation(
    cases: tuple[EvaluationCase, ...],
    observations: tuple[EvaluationObservation, ...],
) -> EvaluationMetrics:
    by_id = {observation.case_id: observation for observation in observations}
    if set(by_id) != {case.case_id for case in cases}:
        raise ValueError("observations must match every evaluation case exactly")

    retrieval_scores: list[float] = []
    citation_scores: list[float] = []
    faithfulness_scores: list[float] = []
    abstention_scores: list[float] = []
    for case in cases:
        observation = by_id[case.case_id]
        relevant = observation.relevant_source_ids
        if relevant:
            retrieved = set(observation.retrieved_source_ids)
            retrieval_scores.append(len(relevant & retrieved) / len(relevant))

        cited = set(observation.cited_source_ids)
        retrieved = set(observation.retrieved_source_ids)
        citation_scores.append(
            1.0
            if (case.should_abstain and not cited)
            or (not case.should_abstain and bool(cited) and cited <= retrieved)
            else 0.0
        )
        faithfulness_scores.append(_faithfulness(case, observation))
        abstention_scores.append(float(observation.abstained == case.should_abstain))

    return EvaluationMetrics(
        retrieval_recall=_mean(retrieval_scores),
        citation_correctness=_mean(citation_scores),
        answer_faithfulness=_mean(faithfulness_scores),
        abstention_accuracy=_mean(abstention_scores),
        case_count=len(cases),
    )


def run_rag_evaluation(
    cases: tuple[EvaluationCase, ...],
    *,
    vector_store: Any,
    model: Any,
    limit: int = 5,
) -> tuple[EvaluationMetrics, tuple[EvaluationObservation, ...]]:
    collection = vector_store.get(include=["metadatas"])
    title_to_ids: dict[str, set[str]] = {}
    for source_id, metadata in zip(
        collection["ids"], collection["metadatas"], strict=True
    ):
        title = str((metadata or {}).get("title") or "")
        if title:
            title_to_ids.setdefault(title, set()).add(str(source_id))

    observations: list[EvaluationObservation] = []
    for case in cases:
        missing_titles = [
            title for title in case.relevant_titles if title not in title_to_ids
        ]
        if missing_titles:
            raise ValueError(
                f"evaluation case {case.case_id} references missing titles: "
                + ", ".join(missing_titles)
            )
        relevant_ids = frozenset(
            source_id
            for title in case.relevant_titles
            for source_id in title_to_ids[title]
        )
        matches = search_reviews(vector_store, case.question, limit=limit)
        result = answer_question(
            case.question,
            vector_store=vector_store,
            model=model,
            limit=limit,
        )
        observations.append(
            EvaluationObservation(
                case_id=case.case_id,
                relevant_source_ids=relevant_ids,
                retrieved_source_ids=tuple(
                    _source_id(match.document) for match in matches
                ),
                cited_source_ids=tuple(
                    _source_id(source.document) for source in result.sources
                ),
                cited_text="\n".join(
                    source.document.page_content for source in result.sources
                ),
                answer=result.answer,
                abstained=result.answer == NO_MATCH_MESSAGE,
            )
        )

    materialized = tuple(observations)
    return score_evaluation(cases, materialized), materialized

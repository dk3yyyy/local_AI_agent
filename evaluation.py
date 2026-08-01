import csv
import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

from agent import answer_question

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
    category: str = "uncategorized"
    gold_source_ids: tuple[str, ...] = ()
    split: str = "test"
    difficulty: str = "unspecified"


@dataclass(frozen=True)
class EvaluationObservation:
    case_id: str
    relevant_source_ids: frozenset[str]
    retrieved_source_ids: tuple[str, ...]
    cited_source_ids: tuple[str, ...]
    cited_text: str
    answer: str
    abstained: bool
    outcome: str = "unknown"
    latency_ms: float | None = None
    raw_model_response: str = ""
    repair_model_response: str | None = None
    initial_failure_reason: str | None = None
    failure_reason: str | None = None
    repair_attempted: bool = False


@dataclass(frozen=True)
class EvaluationMetrics:
    retrieval_recall: float
    citation_validity: float
    reference_term_support_proxy: float
    expected_action_accuracy: float
    case_count: int
    answer_success_rate: float = 0.0
    abstention_recall: float = 0.0
    answerable_case_count: int = 0
    abstention_case_count: int = 0

    @property
    def citation_correctness(self) -> float:
        """Backward-compatible alias; this metric checks citation validity only."""
        return self.citation_validity

    @property
    def answer_faithfulness(self) -> float:
        """Compatibility alias for the earlier, less precise metric name."""
        return self.reference_term_support_proxy

    @property
    def abstention_accuracy(self) -> float:
        """Compatibility alias for the earlier aggregate metric name."""
        return self.expected_action_accuracy

    def as_dict(self) -> dict[str, float | int]:
        return {
            "retrieval_recall": self.retrieval_recall,
            "citation_validity": self.citation_validity,
            "reference_term_support_proxy": self.reference_term_support_proxy,
            "expected_action_accuracy": self.expected_action_accuracy,
            "answer_success_rate": self.answer_success_rate,
            "abstention_recall": self.abstention_recall,
            "case_count": self.case_count,
            "answerable_case_count": self.answerable_case_count,
            "abstention_case_count": self.abstention_case_count,
        }


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_k: float
    hit_rate_at_k: float
    mrr_at_k: float
    evaluated_case_count: int
    limit: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "recall_at_k": self.recall_at_k,
            "hit_rate_at_k": self.hit_rate_at_k,
            "mrr_at_k": self.mrr_at_k,
            "evaluated_case_count": self.evaluated_case_count,
            "limit": self.limit,
        }


_TOKEN_PATTERN = re.compile(r"(?u)\b[^\W_]{2,}\b")


def _tokens(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(_TOKEN_PATTERN.findall(normalized))


class BM25Retriever:
    """Small deterministic BM25 keyword baseline for retrieval comparison."""

    def __init__(
        self,
        documents: Sequence[Any],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not documents:
            raise ValueError("BM25 requires at least one document")
        self._k1 = k1
        self._b = b
        self._source_ids = tuple(_source_id(document) for document in documents)
        if any(not source_id for source_id in self._source_ids):
            raise ValueError("every BM25 document requires a source ID")
        if len(set(self._source_ids)) != len(self._source_ids):
            raise ValueError("BM25 document source IDs must be unique")
        self._term_frequencies = tuple(
            Counter(_tokens(str(document.page_content))) for document in documents
        )
        self._document_lengths = tuple(
            sum(frequencies.values()) for frequencies in self._term_frequencies
        )
        self._average_document_length = sum(self._document_lengths) / len(
            self._document_lengths
        )
        self._document_frequencies = Counter(
            term for frequencies in self._term_frequencies for term in frequencies
        )

    def search(self, query: str, *, limit: int = 5) -> tuple[str, ...]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        query_terms = set(_tokens(query))
        document_count = len(self._source_ids)
        scored: list[tuple[float, str]] = []
        for source_id, frequencies, document_length in zip(
            self._source_ids,
            self._term_frequencies,
            self._document_lengths,
            strict=True,
        ):
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequencies[term]
                inverse_document_frequency = math.log(
                    1
                    + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                length_normalization = self._k1 * (
                    1
                    - self._b
                    + self._b
                    * document_length
                    / max(self._average_document_length, 1.0)
                )
                score += inverse_document_frequency * (
                    frequency * (self._k1 + 1) / (frequency + length_normalization)
                )
            if score > 0:
                scored.append((score, source_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return tuple(source_id for _, source_id in scored[:limit])


def load_evaluation_cases(
    path: str | Path,
    *,
    dataset_path: str | Path | None = None,
) -> tuple[EvaluationCase, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "rag-eval-cases/2.0"
    ):
        raise ValueError("evaluation set must use schema rag-eval-cases/2.0")
    items = payload.get("cases")
    if not isinstance(items, list) or not items:
        raise ValueError("evaluation set requires at least one case")
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        raise TypeError("evaluation set dataset provenance must be an object")
    declared_hash = dataset.get("sha256")
    declared_row_count = dataset.get("row_count")
    if not isinstance(declared_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", declared_hash
    ):
        raise ValueError("evaluation dataset requires a lowercase SHA-256")
    if not isinstance(declared_row_count, int) or declared_row_count < 1:
        raise ValueError("evaluation dataset requires a positive integer row_count")
    if dataset_path is not None:
        from hashlib import sha256

        resolved_dataset = Path(dataset_path)
        actual_hash = sha256(resolved_dataset.read_bytes()).hexdigest()
        if declared_hash != actual_hash:
            raise ValueError("evaluation dataset SHA-256 does not match the manifest")
        with resolved_dataset.open(encoding="utf-8", newline="") as handle:
            actual_row_count = sum(1 for _ in csv.DictReader(handle))
        if declared_row_count != actual_row_count:
            raise ValueError("evaluation dataset row_count does not match the manifest")

    def parse_strings(value: Any, *, field: str) -> tuple[str, ...]:
        if not isinstance(value, list):
            raise TypeError(f"{field} must be an array of strings")
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError(f"{field} must contain only non-empty strings")
        return tuple(item.strip() for item in value)

    def parse_case(item: Any) -> EvaluationCase:
        if not isinstance(item, dict):
            raise TypeError("every evaluation case must be an object")
        action = item.get("expected_action")
        if action not in {"answer", "abstain"}:
            raise ValueError("expected_action must be answer or abstain")
        gold_ids = parse_strings(item.get("gold_source_ids"), field="gold_source_ids")
        source_labels = parse_strings(item.get("source_labels"), field="source_labels")
        raw_facts = item.get("reference_facts")
        if not isinstance(raw_facts, list):
            raise TypeError("reference_facts must be an array")
        facts_list: list[ReferenceFact] = []
        for fact in raw_facts:
            if not isinstance(fact, dict):
                raise TypeError("every reference fact must be an object")
            facts_list.append(
                ReferenceFact(
                    answer_terms=parse_strings(
                        fact.get("answer_terms"), field="answer_terms"
                    ),
                    source_terms=parse_strings(
                        fact.get("source_terms"), field="source_terms"
                    ),
                )
            )
        facts = tuple(facts_list)
        case = EvaluationCase(
            case_id=str(item.get("id") or "").strip(),
            question=str(item.get("question") or "").strip(),
            relevant_titles=source_labels,
            reference_facts=facts,
            should_abstain=action == "abstain",
            category=str(item.get("category") or "").strip(),
            gold_source_ids=gold_ids,
            split=str(item.get("split") or "").strip(),
            difficulty=str(item.get("difficulty") or "").strip(),
        )
        if not case.case_id or not case.question:
            raise ValueError("evaluation cases require non-empty IDs and questions")
        if not case.category or not case.split or not case.difficulty:
            raise ValueError("evaluation cases require category, split, and difficulty")
        if len(set(case.gold_source_ids)) != len(case.gold_source_ids):
            raise ValueError(
                f"evaluation case {case.case_id} has duplicate gold source IDs"
            )
        if case.should_abstain and (
            case.gold_source_ids or case.relevant_titles or case.reference_facts
        ):
            raise ValueError(
                "abstention cases cannot declare gold sources, source labels, "
                "or reference facts"
            )
        if len(case.gold_source_ids) != len(case.relevant_titles):
            raise ValueError(
                f"evaluation case {case.case_id} must label every gold source"
            )
        if any(
            not fact.answer_terms or not fact.source_terms
            for fact in case.reference_facts
        ):
            raise ValueError(
                f"evaluation case {case.case_id} has an empty reference fact"
            )
        if not case.should_abstain and not case.gold_source_ids:
            raise ValueError("answer cases require at least one gold source ID")
        if not case.should_abstain and not case.reference_facts:
            raise ValueError("answer cases require at least one reference fact")
        return case

    cases = tuple(parse_case(item) for item in items)
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("evaluation case IDs must be unique")
    return cases


def _source_id(document: Any) -> str:
    return str(document.metadata.get("source_id") or document.id or "")


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def retrieval_metrics_from_observations(
    cases: tuple[EvaluationCase, ...],
    observations: tuple[EvaluationObservation, ...],
    *,
    limit: int,
) -> RetrievalMetrics:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    by_id = {observation.case_id: observation for observation in observations}
    if len(observations) != len(by_id) or len(observations) != len(cases):
        raise ValueError("observations must contain each evaluation case exactly once")
    if set(by_id) != {case.case_id for case in cases}:
        raise ValueError("observations must match every evaluation case exactly")

    recall_scores: list[float] = []
    hit_scores: list[float] = []
    reciprocal_ranks: list[float] = []
    for case in cases:
        observation = by_id[case.case_id]
        relevant = observation.relevant_source_ids
        if not relevant:
            continue
        retrieved = observation.retrieved_source_ids[:limit]
        retrieved_set = set(retrieved)
        recall_scores.append(len(relevant & retrieved_set) / len(relevant))
        hit_scores.append(float(bool(relevant & retrieved_set)))
        reciprocal_ranks.append(
            next(
                (
                    1.0 / rank
                    for rank, source_id in enumerate(retrieved, start=1)
                    if source_id in relevant
                ),
                0.0,
            )
        )
    return RetrievalMetrics(
        recall_at_k=_mean(recall_scores),
        hit_rate_at_k=_mean(hit_scores),
        mrr_at_k=_mean(reciprocal_ranks),
        evaluated_case_count=len(recall_scores),
        limit=limit,
    )


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
    if not cases:
        raise ValueError("evaluation requires at least one case")
    by_id = {observation.case_id: observation for observation in observations}
    if len(observations) != len(by_id) or len(observations) != len(cases):
        raise ValueError("observations must contain each evaluation case exactly once")
    if set(by_id) != {case.case_id for case in cases}:
        raise ValueError("observations must match every evaluation case exactly")

    retrieval_scores: list[float] = []
    citation_scores: list[float] = []
    support_scores: list[float] = []
    expected_action_scores: list[float] = []
    answer_success_scores: list[float] = []
    abstention_scores: list[float] = []
    for case in cases:
        observation = by_id[case.case_id]
        relevant = observation.relevant_source_ids
        if relevant:
            retrieved = set(observation.retrieved_source_ids)
            retrieval_scores.append(len(relevant & retrieved) / len(relevant))

        cited = set(observation.cited_source_ids)
        retrieved = set(observation.retrieved_source_ids)
        answer_succeeded = observation.outcome in {
            "answered",
            "answered_after_repair",
        } or (
            observation.outcome == "unknown"
            and not observation.abstained
            and bool(cited)
        )
        if case.should_abstain:
            abstention_succeeded = observation.abstained and not cited
            abstention_scores.append(float(abstention_succeeded))
            expected_action_scores.append(float(abstention_succeeded))
        else:
            citation_scores.append(float(bool(cited) and cited <= retrieved))
            support_scores.append(_faithfulness(case, observation))
            answer_success_scores.append(float(answer_succeeded))
            expected_action_scores.append(float(answer_succeeded))

    return EvaluationMetrics(
        retrieval_recall=_mean(retrieval_scores),
        citation_validity=_mean(citation_scores),
        reference_term_support_proxy=_mean(support_scores),
        expected_action_accuracy=_mean(expected_action_scores),
        case_count=len(cases),
        answer_success_rate=_mean(answer_success_scores),
        abstention_recall=_mean(abstention_scores),
        answerable_case_count=len(answer_success_scores),
        abstention_case_count=len(abstention_scores),
    )


def run_rag_evaluation(
    cases: tuple[EvaluationCase, ...],
    *,
    vector_store: Any,
    model: Any,
    limit: int = 5,
) -> tuple[EvaluationMetrics, tuple[EvaluationObservation, ...]]:
    collection = vector_store.get(include=["metadatas"])
    available_source_ids = {str(source_id) for source_id in collection["ids"]}
    title_to_ids: dict[str, set[str]] = {}
    for source_id, metadata in zip(
        collection["ids"], collection["metadatas"], strict=True
    ):
        title = str((metadata or {}).get("title") or "")
        if title:
            title_to_ids.setdefault(title, set()).add(str(source_id))

    observations: list[EvaluationObservation] = []
    for case in cases:
        relevant_ids = _relevant_ids(
            case,
            title_to_ids,
            available_source_ids=available_source_ids,
        )
        started = perf_counter()
        result = answer_question(
            case.question,
            vector_store=vector_store,
            model=model,
            limit=limit,
        )
        latency_ms = (perf_counter() - started) * 1000
        if result.failure_reason == "retrieved_source_missing_id":
            outcome = "retrieved_source_missing_id"
        elif not result.retrieved_source_ids:
            outcome = "empty_retrieval"
        elif result.abstained:
            if not result.repair_attempted:
                outcome = "model_abstention"
            elif (
                result.initial_failure_reason == "clean_abstention"
                and result.failure_reason == "clean_abstention"
            ):
                outcome = "model_abstention_confirmed_after_repair"
            elif result.initial_failure_reason == "clean_abstention":
                outcome = "model_abstention_preserved_after_failed_repair"
            else:
                outcome = "model_abstention_after_repair"
        elif not result.sources:
            outcome = (
                "citation_validation_rejection_after_repair"
                if result.repair_attempted
                else "citation_validation_rejection"
            )
        elif result.repair_attempted:
            outcome = "answered_after_repair"
        else:
            outcome = "answered"
        observations.append(
            EvaluationObservation(
                case_id=case.case_id,
                relevant_source_ids=relevant_ids,
                retrieved_source_ids=result.retrieved_source_ids,
                cited_source_ids=tuple(
                    _source_id(source.document) for source in result.sources
                ),
                cited_text="\n".join(
                    source.document.page_content for source in result.sources
                ),
                answer=result.answer,
                abstained=result.abstained,
                outcome=outcome,
                latency_ms=latency_ms,
                raw_model_response=result.raw_response,
                repair_model_response=result.repair_response,
                initial_failure_reason=result.initial_failure_reason,
                failure_reason=result.failure_reason,
                repair_attempted=result.repair_attempted,
            )
        )

    materialized = tuple(observations)
    return score_evaluation(cases, materialized), materialized


def run_bm25_baseline(
    cases: tuple[EvaluationCase, ...],
    *,
    vector_store: Any,
    limit: int = 5,
) -> tuple[RetrievalMetrics, tuple[EvaluationObservation, ...]]:
    collection = vector_store.get(include=["metadatas", "documents"])
    documents = tuple(
        _document_from_collection(source_id, metadata, content)
        for source_id, metadata, content in zip(
            collection["ids"],
            collection["metadatas"],
            collection["documents"],
            strict=True,
        )
    )
    title_to_ids = _title_to_source_ids(documents)
    available_source_ids = {_source_id(document) for document in documents}
    retriever = BM25Retriever(documents)
    observations = tuple(
        EvaluationObservation(
            case_id=case.case_id,
            relevant_source_ids=_relevant_ids(
                case,
                title_to_ids,
                available_source_ids=available_source_ids,
            ),
            retrieved_source_ids=retriever.search(case.question, limit=limit),
            cited_source_ids=(),
            cited_text="",
            answer="",
            abstained=False,
            outcome="retrieval_only",
        )
        for case in cases
    )
    return (
        retrieval_metrics_from_observations(cases, observations, limit=limit),
        observations,
    )


def _document_from_collection(
    source_id: object,
    metadata: object,
    content: object,
) -> Any:
    from langchain_core.documents import Document

    if metadata is None:
        resolved_metadata = {}
    elif isinstance(metadata, Mapping):
        resolved_metadata = dict(metadata)
    else:
        raise TypeError("collection metadata must be a mapping")
    resolved_metadata.setdefault("source_id", str(source_id))
    return Document(
        id=str(source_id),
        page_content=str(content or ""),
        metadata=resolved_metadata,
    )


def _title_to_source_ids(documents: Sequence[Any]) -> dict[str, set[str]]:
    title_to_ids: dict[str, set[str]] = {}
    for document in documents:
        title = str(document.metadata.get("title") or "")
        if title:
            title_to_ids.setdefault(title, set()).add(_source_id(document))
    return title_to_ids


def _relevant_ids(
    case: EvaluationCase,
    title_to_ids: Mapping[str, set[str]],
    *,
    available_source_ids: set[str],
) -> frozenset[str]:
    if case.gold_source_ids:
        missing_ids = sorted(set(case.gold_source_ids) - available_source_ids)
        if missing_ids:
            raise ValueError(
                f"evaluation case {case.case_id} references missing source IDs: "
                + ", ".join(missing_ids)
            )
        return frozenset(case.gold_source_ids)
    missing_titles = [
        title for title in case.relevant_titles if title not in title_to_ids
    ]
    if missing_titles:
        raise ValueError(
            f"evaluation case {case.case_id} references missing titles: "
            + ", ".join(missing_titles)
        )
    return frozenset(
        source_id for title in case.relevant_titles for source_id in title_to_ids[title]
    )


def _as_mapping(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    converted = value.as_dict()
    if not isinstance(converted, Mapping):
        raise TypeError("metric as_dict() must return a mapping")
    return dict(converted)


def build_evaluation_report(
    *,
    cases: tuple[EvaluationCase, ...],
    rag_metrics: Mapping[str, Any] | EvaluationMetrics,
    semantic_metrics: Mapping[str, Any] | RetrievalMetrics,
    baseline_metrics: Mapping[str, Any] | RetrievalMetrics,
    observations: tuple[EvaluationObservation, ...],
    configuration: Mapping[str, Any],
    provenance: Mapping[str, Any],
    baseline_observations: tuple[EvaluationObservation, ...] = (),
    generated_at: str | None = None,
    include_raw_responses: bool = False,
) -> dict[str, Any]:
    timestamp = generated_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    categories = Counter(case.category for case in cases)
    latencies = [
        observation.latency_ms
        for observation in observations
        if observation.latency_ms is not None
    ]
    outcome_counts = Counter(observation.outcome for observation in observations)
    initial_failure_counts = Counter(
        observation.initial_failure_reason
        for observation in observations
        if observation.initial_failure_reason is not None
    )
    final_failure_counts = Counter(
        observation.failure_reason
        for observation in observations
        if observation.failure_reason is not None
    )
    return {
        "schema_version": 3,
        "generated_at": timestamp,
        "configuration": dict(configuration),
        "provenance": dict(provenance),
        "evaluation_set": {
            "case_count": len(cases),
            "answerable_case_count": sum(not case.should_abstain for case in cases),
            "abstention_case_count": sum(case.should_abstain for case in cases),
            "categories": dict(sorted(categories.items())),
        },
        "results": {
            "semantic_retrieval": _as_mapping(semantic_metrics),
            "bm25_baseline": _as_mapping(baseline_metrics),
            "rag": _as_mapping(rag_metrics),
        },
        "timing": {
            "rag_total_latency_ms": round(sum(latencies), 3),
            "rag_mean_latency_ms": round(_mean(latencies), 3),
            "rag_median_latency_ms": round(median(latencies), 3) if latencies else 0.0,
        },
        "diagnostics": {
            "outcomes": dict(sorted(outcome_counts.items())),
            "initial_failure_reasons": dict(sorted(initial_failure_counts.items())),
            "final_failure_reasons": dict(sorted(final_failure_counts.items())),
            "repair_attempt_count": sum(
                observation.repair_attempted for observation in observations
            ),
            "repair_success_count": outcome_counts["answered_after_repair"],
            "raw_responses_included": include_raw_responses,
        },
        "observations": {
            "rag": [
                _observation_as_dict(
                    observation,
                    include_raw_responses=include_raw_responses,
                )
                for observation in observations
            ],
            "bm25_baseline": [
                _observation_as_dict(observation)
                for observation in baseline_observations
            ],
        },
    }


def _observation_as_dict(
    observation: EvaluationObservation,
    *,
    include_raw_responses: bool = False,
) -> dict[str, Any]:
    serialized = {
        "case_id": observation.case_id,
        "relevant_source_ids": sorted(observation.relevant_source_ids),
        "retrieved_source_ids": list(observation.retrieved_source_ids),
        "cited_source_ids": list(observation.cited_source_ids),
        "answer": observation.answer,
        "abstained": observation.abstained,
        "outcome": observation.outcome,
        "latency_ms": observation.latency_ms,
        "initial_failure_reason": observation.initial_failure_reason,
        "failure_reason": observation.failure_reason,
        "repair_attempted": observation.repair_attempted,
    }
    if include_raw_responses:
        serialized["raw_model_response"] = observation.raw_model_response
        serialized["repair_model_response"] = observation.repair_model_response
    return serialized


def _metric(value: object) -> str:
    if not isinstance(value, (str, int, float)):
        raise TypeError("metric value must be numeric")
    return f"{float(value):.3f}"


def _report_markdown(report: Mapping[str, Any]) -> str:
    results = report["results"]
    semantic = results["semantic_retrieval"]
    baseline = results["bm25_baseline"]
    rag = results["rag"]
    evaluation_set = report["evaluation_set"]
    configuration = report["configuration"]
    provenance = report["provenance"]
    timing = report["timing"]
    diagnostics = report.get(
        "diagnostics",
        {
            "outcomes": {},
            "initial_failure_reasons": {},
            "final_failure_reasons": {},
            "repair_attempt_count": 0,
            "repair_success_count": 0,
            "raw_responses_included": False,
        },
    )
    return "\n".join(
        (
            "# Evaluation report",
            "",
            f"Generated: `{report['generated_at']}`",
            "",
            "## Scope",
            "",
            (
                f"- Cases: **{evaluation_set['case_count']}** "
                f"({evaluation_set['answerable_case_count']} answerable, "
                f"{evaluation_set['abstention_case_count']} abstention)"
            ),
            f"- Chat model: `{configuration.get('chat_model', 'unknown')}`",
            f"- Embedding model: `{configuration.get('embedding_model', 'unknown')}`",
            f"- Ollama runtime: `{configuration.get('ollama_version', 'unknown')}`",
            f"- Evidence limit: **{semantic['limit']}**",
            f"- Dataset SHA-256: `{provenance.get('dataset_sha256', 'unknown')}`",
            f"- Cases SHA-256: `{provenance.get('cases_sha256', 'unknown')}`",
            "",
            "## Retrieval comparison",
            "",
            "| Retriever | Recall@k | Hit rate@k | MRR@k |",
            "| --- | ---: | ---: | ---: |",
            (
                f"| Semantic | {_metric(semantic['recall_at_k'])} | "
                f"{_metric(semantic['hit_rate_at_k'])} | "
                f"{_metric(semantic['mrr_at_k'])} |"
            ),
            (
                f"| BM25 keyword baseline | {_metric(baseline['recall_at_k'])} | "
                f"{_metric(baseline['hit_rate_at_k'])} | "
                f"{_metric(baseline['mrr_at_k'])} |"
            ),
            "",
            "## Model-dependent results",
            "",
            "| Metric | Score |",
            "| --- | ---: |",
            f"| Citation validity | {_metric(rag['citation_validity'])} |",
            (
                "| Reference-term support proxy | "
                f"{_metric(rag['reference_term_support_proxy'])} |"
            ),
            f"| Expected-action accuracy | {_metric(rag['expected_action_accuracy'])} |",
            f"| Answer success (answerable cases) | {_metric(rag['answer_success_rate'])} |",
            f"| Abstention recall (abstention cases) | {_metric(rag['abstention_recall'])} |",
            "",
            "## Answer diagnostics",
            "",
            f"- Outcomes: `{json.dumps(diagnostics['outcomes'], sort_keys=True)}`",
            (
                "- Initial validation reasons: `"
                f"{json.dumps(diagnostics['initial_failure_reasons'], sort_keys=True)}`"
            ),
            (
                "- Final non-success reasons: `"
                f"{json.dumps(diagnostics['final_failure_reasons'], sort_keys=True)}`"
            ),
            f"- Repair attempts: **{diagnostics['repair_attempt_count']}**",
            f"- Successful repairs: **{diagnostics['repair_success_count']}**",
            "",
            "## Timing",
            "",
            f"- Total RAG latency: **{timing['rag_total_latency_ms'] / 1000:.2f}s**",
            f"- Mean per case: **{timing['rag_mean_latency_ms']:.1f}ms**",
            f"- Median per case: **{timing['rag_median_latency_ms']:.1f}ms**",
            "",
            (
                "These scores describe this fixed dataset, case set, retrieval limit, "
                "and local model configuration. Relevance judgments are known-positive, "
                "not exhaustive. The reference-term support score is a transparent "
                "heuristic, not an LLM judge or a general factuality guarantee."
            ),
            "",
        )
    )


def write_evaluation_report(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    resolved_json = Path(json_path)
    resolved_markdown = Path(markdown_path)
    resolved_json.parent.mkdir(parents=True, exist_ok=True)
    resolved_markdown.parent.mkdir(parents=True, exist_ok=True)
    resolved_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    resolved_markdown.write_text(_report_markdown(report), encoding="utf-8")

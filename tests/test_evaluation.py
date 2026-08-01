import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from agent import AnswerResult
from evaluation import (
    DEFAULT_EVALUATION_PATH,
    EvaluationCase,
    EvaluationObservation,
    ReferenceFact,
    load_evaluation_cases,
    run_rag_evaluation,
    score_evaluation,
)


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class EvaluationModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> FakeResponse:
        self.prompts.append(prompt)
        return FakeResponse(self.response)


class EvaluationSequenceModel:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> FakeResponse:
        self.prompts.append(prompt)
        return FakeResponse(self.responses.pop(0))


class EvaluationStore:
    def __init__(self, *, return_matches: bool = True) -> None:
        self.search_calls = 0
        self.document = Document(
            page_content="The crust was perfectly crispy.",
            metadata={"source_id": "review-a", "title": "Best pizza"},
            id="review-a",
        )
        self.return_matches = return_matches

    def get(self, **_: object) -> dict[str, list[object]]:
        return {
            "ids": ["review-a"],
            "metadatas": [self.document.metadata],
        }

    def similarity_search_with_score(
        self, query: str, *, k: int
    ) -> list[tuple[Document, float]]:
        self.search_calls += 1
        if not self.return_matches:
            return []
        return [(self.document, 0.9)]


class RAGEvaluationTest(unittest.TestCase):
    def test_loads_curated_evaluation_set(self) -> None:
        cases = load_evaluation_cases(DEFAULT_EVALUATION_PATH)

        self.assertEqual(len(cases), 30)
        self.assertEqual(sum(case.should_abstain for case in cases), 5)
        self.assertTrue(all(case.category for case in cases))
        self.assertTrue(any(case.reference_facts for case in cases))

    def test_scores_retrieval_citations_faithfulness_and_abstention(self) -> None:
        cases = (
            EvaluationCase(
                case_id="answer",
                question="Is the crust crisp?",
                relevant_titles=("Best pizza",),
                reference_facts=(
                    ReferenceFact(
                        answer_terms=("crispy",),
                        source_terms=("perfectly crispy",),
                    ),
                ),
            ),
            EvaluationCase(
                case_id="abstain",
                question="Is parking available?",
                relevant_titles=(),
                reference_facts=(),
                should_abstain=True,
            ),
        )
        observations = (
            EvaluationObservation(
                case_id="answer",
                relevant_source_ids=frozenset({"review-a"}),
                retrieved_source_ids=("review-a", "review-b"),
                cited_source_ids=("review-a",),
                cited_text="The crust was perfectly crispy.",
                answer="Guests describe it as crispy [1].",
                abstained=False,
            ),
            EvaluationObservation(
                case_id="abstain",
                relevant_source_ids=frozenset(),
                retrieved_source_ids=("review-b",),
                cited_source_ids=(),
                cited_text="",
                answer="I could not find matching evidence.",
                abstained=True,
            ),
        )

        metrics = score_evaluation(cases, observations)

        self.assertEqual(metrics.retrieval_recall, 1.0)
        self.assertEqual(metrics.citation_correctness, 1.0)
        self.assertEqual(metrics.answer_faithfulness, 1.0)
        self.assertEqual(metrics.abstention_accuracy, 1.0)
        self.assertEqual(metrics.case_count, 2)

    def test_pipeline_reuses_the_exact_answer_retrieval(self) -> None:
        case = EvaluationCase(
            case_id="answer",
            question="Is the crust crisp?",
            relevant_titles=("Best pizza",),
            reference_facts=(
                ReferenceFact(
                    answer_terms=("crispy",),
                    source_terms=("perfectly crispy",),
                ),
            ),
        )
        store = EvaluationStore()
        model = EvaluationModel("Guests describe it as crispy [review-a].")

        metrics, observations = run_rag_evaluation(
            (case,), vector_store=store, model=model
        )

        self.assertEqual(store.search_calls, 1)
        self.assertEqual(observations[0].retrieved_source_ids, ("review-a",))
        self.assertEqual(observations[0].cited_source_ids, ("review-a",))
        self.assertEqual(metrics.citation_correctness, 1.0)

    def test_empty_retrieval_is_not_counted_as_model_abstention(self) -> None:
        case = EvaluationCase(
            case_id="abstain",
            question="Is parking available?",
            relevant_titles=(),
            reference_facts=(),
            should_abstain=True,
        )
        store = EvaluationStore(return_matches=False)
        model = EvaluationModel("INSUFFICIENT_EVIDENCE")

        metrics, observations = run_rag_evaluation(
            (case,), vector_store=store, model=model
        )

        self.assertEqual(store.search_calls, 1)
        self.assertFalse(observations[0].abstained)
        self.assertEqual(model.prompts, [])
        self.assertEqual(metrics.abstention_accuracy, 0.0)

    def test_missing_source_id_is_classified_as_data_integrity_failure(self) -> None:
        case = EvaluationCase(
            case_id="answer",
            question="Is the crust crisp?",
            relevant_titles=("Best pizza",),
            reference_facts=(),
        )
        malformed = AnswerResult(
            answer="I could not produce an answer with valid citations.",
            sources=(),
            failure_reason="retrieved_source_missing_id",
        )

        with patch("evaluation.answer_question", return_value=malformed):
            _, observations = run_rag_evaluation(
                (case,), vector_store=EvaluationStore(), model=object()
            )

        self.assertEqual(observations[0].outcome, "retrieved_source_missing_id")

    def test_pipeline_preserves_raw_rejection_and_repair_diagnostics(self) -> None:
        case = EvaluationCase(
            case_id="answer",
            question="Is the crust crisp?",
            relevant_titles=("Best pizza",),
            reference_facts=(
                ReferenceFact(
                    answer_terms=("crispy",),
                    source_terms=("perfectly crispy",),
                ),
            ),
        )
        model = EvaluationSequenceModel(
            "The crust is crispy.",
            "The crust is crispy [1].",
        )

        metrics, observations = run_rag_evaluation(
            (case,), vector_store=EvaluationStore(), model=model
        )

        observation = observations[0]
        self.assertEqual(observation.outcome, "answered_after_repair")
        self.assertEqual(observation.raw_model_response, "The crust is crispy.")
        self.assertEqual(observation.repair_model_response, "The crust is crispy [1].")
        self.assertEqual(observation.initial_failure_reason, "missing_citations")
        self.assertIsNone(observation.failure_reason)
        self.assertTrue(observation.repair_attempted)
        self.assertEqual(metrics.answer_success_rate, 1.0)

    def test_pipeline_labels_an_abstention_returned_by_repair(self) -> None:
        case = EvaluationCase(
            case_id="abstain",
            question="Is parking available?",
            relevant_titles=(),
            reference_facts=(),
            should_abstain=True,
        )
        model = EvaluationSequenceModel(
            "Parking is available.",
            "INSUFFICIENT_EVIDENCE",
        )

        metrics, observations = run_rag_evaluation(
            (case,), vector_store=EvaluationStore(), model=model
        )

        self.assertEqual(observations[0].outcome, "model_abstention_after_repair")
        self.assertTrue(observations[0].repair_attempted)
        self.assertEqual(observations[0].failure_reason, "clean_abstention")
        self.assertEqual(metrics.abstention_recall, 1.0)

    def test_pipeline_labels_a_preserved_initial_abstention_truthfully(self) -> None:
        case = EvaluationCase(
            case_id="abstain",
            question="Is parking available?",
            relevant_titles=(),
            reference_facts=(),
            should_abstain=True,
        )
        model = EvaluationSequenceModel(
            "INSUFFICIENT_EVIDENCE",
            "Unsupported parking claim [9].",
        )

        metrics, observations = run_rag_evaluation(
            (case,), vector_store=EvaluationStore(), model=model
        )

        observation = observations[0]
        self.assertEqual(
            observation.outcome,
            "model_abstention_preserved_after_failed_repair",
        )
        self.assertEqual(observation.initial_failure_reason, "clean_abstention")
        self.assertEqual(observation.failure_reason, "out_of_range_citation")
        self.assertEqual(metrics.abstention_recall, 1.0)

    def test_pipeline_labels_an_abstention_confirmed_by_repair(self) -> None:
        case = EvaluationCase(
            case_id="abstain",
            question="Is parking available?",
            relevant_titles=(),
            reference_facts=(),
            should_abstain=True,
        )
        model = EvaluationSequenceModel(
            "INSUFFICIENT_EVIDENCE",
            "INSUFFICIENT_EVIDENCE",
        )

        metrics, observations = run_rag_evaluation(
            (case,), vector_store=EvaluationStore(), model=model
        )

        self.assertEqual(
            observations[0].outcome,
            "model_abstention_confirmed_after_repair",
        )
        self.assertEqual(metrics.abstention_recall, 1.0)

    def test_penalizes_missing_retrieval_invalid_citation_and_false_answer(
        self,
    ) -> None:
        case = EvaluationCase(
            case_id="bad",
            question="Is the crust crisp?",
            relevant_titles=("Best pizza",),
            reference_facts=(
                ReferenceFact(
                    answer_terms=("crispy",),
                    source_terms=("perfectly crispy",),
                ),
            ),
        )
        observation = EvaluationObservation(
            case_id="bad",
            relevant_source_ids=frozenset({"review-a"}),
            retrieved_source_ids=("review-b",),
            cited_source_ids=("review-c",),
            cited_text="Unrelated evidence.",
            answer="It is soggy [1].",
            abstained=True,
        )

        metrics = score_evaluation((case,), (observation,))

        self.assertEqual(metrics.retrieval_recall, 0.0)
        self.assertEqual(metrics.citation_correctness, 0.0)
        self.assertEqual(metrics.answer_faithfulness, 0.0)
        self.assertEqual(metrics.abstention_accuracy, 0.0)

    def test_rejection_is_not_a_successful_answer_or_support_score(self) -> None:
        answer_case = EvaluationCase(
            case_id="answer",
            question="Is the crust crisp?",
            relevant_titles=("Best pizza",),
            reference_facts=(
                ReferenceFact(
                    answer_terms=("crispy",),
                    source_terms=("crispy",),
                ),
            ),
        )
        abstain_case = EvaluationCase(
            case_id="abstain",
            question="Is parking available?",
            relevant_titles=(),
            reference_facts=(),
            should_abstain=True,
        )
        rejected = EvaluationObservation(
            case_id="answer",
            relevant_source_ids=frozenset({"review-a"}),
            retrieved_source_ids=("review-a",),
            cited_source_ids=(),
            cited_text="",
            answer="I could not produce an answer with valid citations.",
            abstained=False,
            outcome="citation_validation_rejection",
        )
        abstained = EvaluationObservation(
            case_id="abstain",
            relevant_source_ids=frozenset(),
            retrieved_source_ids=("review-a",),
            cited_source_ids=(),
            cited_text="",
            answer="I could not find matching evidence.",
            abstained=True,
            outcome="model_abstention",
        )

        metrics = score_evaluation((answer_case, abstain_case), (rejected, abstained))

        self.assertEqual(metrics.expected_action_accuracy, 0.5)
        self.assertEqual(metrics.answer_success_rate, 0.0)
        self.assertEqual(metrics.abstention_recall, 1.0)
        self.assertEqual(metrics.reference_term_support_proxy, 0.0)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            score_evaluation((answer_case,), (rejected, rejected))


if __name__ == "__main__":
    unittest.main()

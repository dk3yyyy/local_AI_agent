import unittest

from evaluation import (
    DEFAULT_EVALUATION_PATH,
    EvaluationCase,
    EvaluationObservation,
    ReferenceFact,
    load_evaluation_cases,
    score_evaluation,
)


class RAGEvaluationTest(unittest.TestCase):
    def test_loads_curated_evaluation_set(self) -> None:
        cases = load_evaluation_cases(DEFAULT_EVALUATION_PATH)

        self.assertEqual(len(cases), 4)
        self.assertTrue(any(case.should_abstain for case in cases))
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


if __name__ == "__main__":
    unittest.main()

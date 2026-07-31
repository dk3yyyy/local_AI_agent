import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from evaluation import (
    BM25Retriever,
    EvaluationCase,
    EvaluationObservation,
    build_evaluation_report,
    load_evaluation_cases,
    retrieval_metrics_from_observations,
    score_evaluation,
    write_evaluation_report,
)


class BenchmarkTest(unittest.TestCase):
    def test_bm25_ranks_exact_keyword_evidence_above_unrelated_text(self) -> None:
        documents = (
            Document(
                id="vegan",
                page_content="Title: Vegan option\nReview: House-made cashew cheese melts well.",
                metadata={"source_id": "vegan", "title": "Vegan option"},
            ),
            Document(
                id="delivery",
                page_content="Title: Slow delivery\nReview: The pizza arrived cold after two hours.",
                metadata={"source_id": "delivery", "title": "Slow delivery"},
            ),
        )

        retriever = BM25Retriever(documents)
        ranked = retriever.search("Does the vegan pizza use cashew cheese?", limit=2)

        self.assertEqual(ranked[0], "vegan")
        self.assertEqual(set(ranked), {"vegan", "delivery"})

    def test_bm25_normalizes_unicode_and_does_not_pad_zero_overlap(self) -> None:
        documents = (
            Document(
                id="crispy",
                page_content="ＣＲＩＳＰＹ crust",
                metadata={"source_id": "crispy"},
            ),
            Document(
                id="other",
                page_content="quiet dining room",
                metadata={"source_id": "other"},
            ),
        )
        retriever = BM25Retriever(documents)

        self.assertEqual(retriever.search("crispy", limit=5), ("crispy",))
        self.assertEqual(retriever.search("wheelchair parking", limit=5), ())

    def test_empty_evaluation_cannot_report_perfect_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one case"):
            score_evaluation((), ())

    def test_retrieval_metrics_report_recall_hit_rate_and_reciprocal_rank(self) -> None:
        cases = (
            EvaluationCase(
                case_id="first",
                question="first",
                relevant_titles=("A",),
                reference_facts=(),
            ),
            EvaluationCase(
                case_id="second",
                question="second",
                relevant_titles=("B", "C"),
                reference_facts=(),
            ),
            EvaluationCase(
                case_id="abstain",
                question="unknown",
                relevant_titles=(),
                reference_facts=(),
                should_abstain=True,
            ),
        )
        observations = (
            EvaluationObservation(
                case_id="first",
                relevant_source_ids=frozenset({"a"}),
                retrieved_source_ids=("x", "a"),
                cited_source_ids=(),
                cited_text="",
                answer="",
                abstained=False,
            ),
            EvaluationObservation(
                case_id="second",
                relevant_source_ids=frozenset({"b", "c"}),
                retrieved_source_ids=("b", "x"),
                cited_source_ids=(),
                cited_text="",
                answer="",
                abstained=False,
            ),
            EvaluationObservation(
                case_id="abstain",
                relevant_source_ids=frozenset(),
                retrieved_source_ids=("x",),
                cited_source_ids=(),
                cited_text="",
                answer="",
                abstained=True,
            ),
        )

        metrics = retrieval_metrics_from_observations(cases, observations, limit=2)

        self.assertEqual(metrics.evaluated_case_count, 2)
        self.assertEqual(metrics.limit, 2)
        self.assertAlmostEqual(metrics.recall_at_k, 0.75)
        self.assertAlmostEqual(metrics.hit_rate_at_k, 1.0)
        self.assertAlmostEqual(metrics.mrr_at_k, 0.75)

    def test_retrieval_metrics_reject_duplicate_observations(self) -> None:
        case = EvaluationCase(
            case_id="one",
            question="one",
            relevant_titles=("One",),
            reference_facts=(),
        )
        observation = EvaluationObservation(
            case_id="one",
            relevant_source_ids=frozenset({"one"}),
            retrieved_source_ids=("one",),
            cited_source_ids=(),
            cited_text="",
            answer="",
            abstained=False,
        )

        with self.assertRaisesRegex(ValueError, "exactly once"):
            retrieval_metrics_from_observations(
                (case,), (observation, observation), limit=1
            )

    def test_report_is_machine_readable_and_writes_matching_markdown(self) -> None:
        cases = (
            EvaluationCase(
                case_id="answer",
                question="Is it crisp?",
                relevant_titles=("Crisp",),
                reference_facts=(),
                category="quality",
            ),
            EvaluationCase(
                case_id="abstain",
                question="Is there parking?",
                relevant_titles=(),
                reference_facts=(),
                should_abstain=True,
                category="abstention",
            ),
        )
        report = build_evaluation_report(
            cases=cases,
            rag_metrics={
                "retrieval_recall": 1.0,
                "citation_validity": 1.0,
                "reference_term_support_proxy": 1.0,
                "expected_action_accuracy": 1.0,
                "answer_success_rate": 1.0,
                "abstention_recall": 1.0,
                "case_count": 2,
            },
            semantic_metrics={
                "recall_at_k": 1.0,
                "hit_rate_at_k": 1.0,
                "mrr_at_k": 1.0,
                "evaluated_case_count": 1,
                "limit": 5,
            },
            baseline_metrics={
                "recall_at_k": 0.0,
                "hit_rate_at_k": 0.0,
                "mrr_at_k": 0.0,
                "evaluated_case_count": 1,
                "limit": 5,
            },
            observations=(),
            configuration={"chat_model": "llama3.2", "embedding_model": "mxbai"},
            provenance={"dataset_sha256": "abc", "cases_sha256": "def"},
            generated_at="2026-07-31T00:00:00Z",
        )

        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(report["evaluation_set"]["case_count"], 2)
        self.assertEqual(report["evaluation_set"]["abstention_case_count"], 1)
        self.assertEqual(report["results"]["bm25_baseline"]["recall_at_k"], 0.0)

        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "report.json"
            markdown_path = Path(directory) / "report.md"
            write_evaluation_report(
                report,
                json_path=json_path,
                markdown_path=markdown_path,
            )

            written = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(written, report)
        self.assertIn("BM25 keyword baseline", markdown)
        self.assertIn("Model-dependent results", markdown)
        self.assertIn("2026-07-31T00:00:00Z", markdown)


class EvaluationSetQualityTest(unittest.TestCase):
    def test_bundled_set_has_broad_unique_coverage(self) -> None:
        from evaluation import DEFAULT_EVALUATION_PATH

        cases = load_evaluation_cases(DEFAULT_EVALUATION_PATH)

        self.assertGreaterEqual(len(cases), 30)
        self.assertEqual(len({case.case_id for case in cases}), len(cases))
        self.assertGreaterEqual(sum(case.should_abstain for case in cases), 5)
        self.assertGreaterEqual(len({case.category for case in cases}), 6)
        self.assertTrue(all(case.category for case in cases))

    def test_manifest_uses_immutable_gold_ids(self) -> None:
        from evaluation import DEFAULT_EVALUATION_PATH

        cases = load_evaluation_cases(DEFAULT_EVALUATION_PATH)
        answerable = [case for case in cases if not case.should_abstain]

        self.assertTrue(answerable)
        self.assertTrue(
            all(
                source_id.startswith("review_")
                for case in answerable
                for source_id in case.gold_source_ids
            )
        )

    def test_loader_rejects_empty_case_sets(self) -> None:
        from evaluation import DEFAULT_EVALUATION_PATH

        payload = json.loads(DEFAULT_EVALUATION_PATH.read_text(encoding="utf-8"))
        payload["cases"] = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "at least one case"):
                load_evaluation_cases(path)

    def test_loader_rejects_contradictory_abstention_cases(self) -> None:
        from evaluation import DEFAULT_EVALUATION_PATH

        payload = json.loads(DEFAULT_EVALUATION_PATH.read_text(encoding="utf-8"))
        payload["cases"][0]["expected_action"] = "abstain"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "abstention cases"):
                load_evaluation_cases(path)

    def test_loader_rejects_string_term_lists_and_empty_terms(self) -> None:
        from evaluation import DEFAULT_EVALUATION_PATH

        payload = json.loads(DEFAULT_EVALUATION_PATH.read_text(encoding="utf-8"))
        payload["cases"][0]["reference_facts"][0]["answer_terms"] = "crispy"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(TypeError, "array of strings"):
                load_evaluation_cases(path)

        payload = json.loads(DEFAULT_EVALUATION_PATH.read_text(encoding="utf-8"))
        payload["cases"][0]["reference_facts"][0]["source_terms"] = [""]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-empty strings"):
                load_evaluation_cases(path)

    def test_loader_verifies_dataset_row_count(self) -> None:
        from evaluation import DEFAULT_EVALUATION_PATH
        from vector import DEFAULT_DATA_PATH

        payload = json.loads(DEFAULT_EVALUATION_PATH.read_text(encoding="utf-8"))
        payload["dataset"]["row_count"] += 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "row_count"):
                load_evaluation_cases(path, dataset_path=DEFAULT_DATA_PATH)


if __name__ == "__main__":
    unittest.main()

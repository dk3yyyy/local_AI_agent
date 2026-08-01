import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from httpx import HTTPError

from evaluation import (
    EvaluationCase,
    EvaluationMetrics,
    EvaluationObservation,
    RetrievalMetrics,
)
from main import _safe_endpoint, build_parser, main, run
from ollama_health import OllamaHealth


class EvaluationCLIReportTest(unittest.TestCase):
    def test_evaluate_parser_accepts_report_directory(self) -> None:
        arguments = build_parser().parse_args(
            ["evaluate", "--report-dir", "docs/evaluation"]
        )

        self.assertEqual(arguments.report_dir, Path("docs/evaluation"))
        self.assertFalse(arguments.include_raw_responses)

        opted_in = build_parser().parse_args(["evaluate", "--include-raw-responses"])
        self.assertTrue(opted_in.include_raw_responses)

    def test_safe_endpoint_removes_credentials_query_and_fragment(self) -> None:
        endpoint = _safe_endpoint(
            "https://user:secret@example.com:11434/api?token=x#part"
        )

        self.assertEqual(endpoint, "https://example.com:11434/api")

    def test_evaluate_writes_json_and_markdown_reports(self) -> None:
        case = EvaluationCase(
            case_id="answer",
            question="Is it crisp?",
            relevant_titles=("Crisp",),
            reference_facts=(),
            category="quality",
        )
        observation = EvaluationObservation(
            case_id="answer",
            relevant_source_ids=frozenset({"a"}),
            retrieved_source_ids=("a",),
            cited_source_ids=("a",),
            cited_text="crisp",
            answer="It is crisp [1].",
            abstained=False,
            raw_model_response="private source-derived text",
            repair_model_response="private repair text",
        )
        rag_metrics = EvaluationMetrics(1.0, 1.0, 1.0, 1.0, 1)
        retrieval_metrics = RetrievalMetrics(1.0, 1.0, 1.0, 1, 5)
        health = OllamaHealth(True, ("chat:latest", "embed:latest"), ())

        with tempfile.TemporaryDirectory() as directory:
            report_dir = Path(directory) / "evaluation"
            arguments = build_parser().parse_args(
                [
                    "evaluate",
                    "--chat-model",
                    "chat",
                    "--embedding-model",
                    "embed",
                    "--report-dir",
                    str(report_dir),
                ]
            )
            with (
                patch("main.load_reviews", return_value=[object()]),
                patch("main._health_for_arguments", return_value=health),
                patch("main._create_runtime", return_value=(object(), object())),
                patch("main.load_evaluation_cases", return_value=(case,)),
                patch(
                    "main.run_rag_evaluation",
                    return_value=(rag_metrics, (observation,)),
                ),
                patch(
                    "main.run_bm25_baseline",
                    return_value=(retrieval_metrics, (observation,)),
                ),
                patch(
                    "main.model_metadata",
                    return_value={
                        "chat": {
                            "resolved_name": "chat:latest",
                            "digest": "sha256:chat",
                            "size": 1,
                            "modified_at": None,
                        },
                        "embed": {
                            "resolved_name": "embed:latest",
                            "digest": "sha256:embed",
                            "size": 2,
                            "modified_at": None,
                        },
                    },
                ),
                patch("main.ollama_version", return_value="0.32.5"),
            ):
                exit_code = run(arguments)
                opted_in_dir = Path(directory) / "evaluation-with-raw"
                opted_in_arguments = build_parser().parse_args(
                    [
                        "evaluate",
                        "--chat-model",
                        "chat",
                        "--embedding-model",
                        "embed",
                        "--report-dir",
                        str(opted_in_dir),
                        "--include-raw-responses",
                    ]
                )
                opted_in_exit_code = run(opted_in_arguments)

            payload = json.loads(
                (report_dir / "evaluation-report.json").read_text(encoding="utf-8")
            )
            markdown = (report_dir / "README.md").read_text(encoding="utf-8")
            opted_in_payload = json.loads(
                (opted_in_dir / "evaluation-report.json").read_text(encoding="utf-8")
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(opted_in_exit_code, 0)
        self.assertEqual(
            payload["configuration"]["models"]["chat"]["digest"], "sha256:chat"
        )
        self.assertEqual(payload["configuration"]["ollama_version"], "0.32.5")
        self.assertEqual(payload["results"]["semantic_retrieval"]["mrr_at_k"], 1.0)
        self.assertNotIn("raw_model_response", payload["observations"]["rag"][0])
        self.assertNotIn("repair_model_response", payload["observations"]["rag"][0])
        self.assertEqual(
            opted_in_payload["observations"]["rag"][0]["raw_model_response"],
            "private source-derived text",
        )
        self.assertEqual(
            opted_in_payload["observations"]["rag"][0]["repair_model_response"],
            "private repair text",
        )
        self.assertIn("Retrieval comparison", markdown)

    def test_evaluate_reports_late_ollama_provenance_failures(self) -> None:
        case = EvaluationCase(
            case_id="answer",
            question="Is it crisp?",
            relevant_titles=("Crisp",),
            reference_facts=(),
            category="quality",
        )
        observation = EvaluationObservation(
            case_id="answer",
            relevant_source_ids=frozenset({"a"}),
            retrieved_source_ids=("a",),
            cited_source_ids=("a",),
            cited_text="crisp",
            answer="It is crisp [1].",
            abstained=False,
        )
        rag_metrics = EvaluationMetrics(1.0, 1.0, 1.0, 1.0, 1)
        retrieval_metrics = RetrievalMetrics(1.0, 1.0, 1.0, 1, 5)
        health = OllamaHealth(True, ("chat:latest", "embed:latest"), ())
        failures = (
            ("ollama_version", HTTPError("Ollama stopped")),
            ("model_metadata", OSError("Ollama stopped")),
        )

        for target, error in failures:
            with (
                self.subTest(target=target),
                tempfile.TemporaryDirectory() as directory,
            ):
                report_dir = Path(directory) / "evaluation"
                stderr = StringIO()
                stdout = StringIO()
                patches = {
                    "ollama_version": patch(
                        "main.ollama_version", return_value="0.32.5"
                    ),
                    "model_metadata": patch("main.model_metadata", return_value={}),
                }
                patches[target] = patch(f"main.{target}", side_effect=error)
                with (
                    patch("main.load_reviews", return_value=[object()]),
                    patch("main._health_for_arguments", return_value=health),
                    patch("main._create_runtime", return_value=(object(), object())),
                    patch("main.load_evaluation_cases", return_value=(case,)),
                    patch(
                        "main.run_rag_evaluation",
                        return_value=(rag_metrics, (observation,)),
                    ),
                    patch(
                        "main.run_bm25_baseline",
                        return_value=(retrieval_metrics, (observation,)),
                    ),
                    patches["ollama_version"],
                    patches["model_metadata"],
                    redirect_stderr(stderr),
                    redirect_stdout(stdout),
                    self.assertRaises(SystemExit) as raised,
                ):
                    main(["evaluate", "--report-dir", str(report_dir)])

                self.assertEqual(raised.exception.code, 2)
                self.assertIn(
                    "could not collect Ollama report metadata", stderr.getvalue()
                )
                self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

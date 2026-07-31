import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Sequence
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from agent import answer_question, create_chat_model
from evaluation import (
    DEFAULT_EVALUATION_PATH,
    build_evaluation_report,
    load_evaluation_cases,
    retrieval_metrics_from_observations,
    run_bm25_baseline,
    run_rag_evaluation,
    write_evaluation_report,
)
from local_ai_agent import __version__
from ollama_health import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_HOST,
    check_ollama,
    model_metadata,
    ollama_version,
)
from vector import (
    DEFAULT_DATA_PATH,
    ReviewDataError,
    create_vector_store,
    load_reviews,
)


def _date_argument(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "use an ISO date such as 2024-03-20"
        ) from error


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--database",
        type=Path,
        help="override the automatically isolated Chroma database path",
    )
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)


def _add_search_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--min-rating", type=int, choices=range(1, 6))
    parser.add_argument("--max-rating", type=int, choices=range(1, 6))
    parser.add_argument("--start-date", type=_date_argument)
    parser.add_argument("--end-date", type=_date_argument)
    parser.add_argument("--sentiment", action="append", default=[])
    parser.add_argument("--restaurant", action="append", default=[])
    parser.add_argument("--country", action="append", default=[])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask grounded questions about local restaurant reviews."
    )
    subparsers = parser.add_subparsers(dest="command")

    status_parser = subparsers.add_parser(
        "status", help="Check data and Ollama readiness"
    )
    _add_runtime_arguments(status_parser)

    ask_parser = subparsers.add_parser("ask", help="Answer one question")
    ask_parser.add_argument("question")
    _add_runtime_arguments(ask_parser)
    _add_search_arguments(ask_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Measure the bundled RAG evaluation set"
    )
    evaluate_parser.add_argument("--cases", type=Path, default=DEFAULT_EVALUATION_PATH)
    evaluate_parser.add_argument("--limit", type=int, default=5)
    evaluate_parser.add_argument(
        "--report-dir",
        type=Path,
        help="write evaluation-report.json and README.md to this directory",
    )
    _add_runtime_arguments(evaluate_parser)

    chat_parser = subparsers.add_parser("chat", help="Start the interactive terminal")
    _add_runtime_arguments(chat_parser)
    _add_search_arguments(chat_parser)
    return parser


def _health_for_arguments(arguments: argparse.Namespace):
    return check_ollama(
        required_models=(arguments.chat_model, arguments.embedding_model),
        host=arguments.ollama_host,
    )


def _print_health(health) -> None:
    if health.ok:
        print("Ollama: ready")
        print("Models: " + ", ".join(health.available_models))
        return
    print("Ollama: not ready")
    if health.error:
        print(f"Error: {health.error}")
    print(health.instructions)


def _print_answer(result) -> None:
    print(f"\n{result.answer}\n")
    if not result.sources:
        return
    print("Sources")
    for source in result.sources:
        metadata = source.document.metadata
        details = []
        for key in ("restaurant", "country", "sentiment", "rating", "date"):
            if key in metadata:
                suffix = "/5" if key == "rating" else ""
                details.append(f"{metadata[key]}{suffix}")
        print(f"[{source.citation_number}] " + " · ".join(details))
        print(source.document.page_content.replace("\n", " | "))


def _safe_endpoint(value: str) -> str:
    """Return an endpoint suitable for a committed report without credentials."""
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.hostname:
        return value.split("?", 1)[0].split("#", 1)[0]
    hostname = parsed.hostname
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance() -> dict[str, str | bool | None]:
    root = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}


def _dependency_versions() -> dict[str, str]:
    resolved: dict[str, str] = {}
    for package in ("chromadb", "langchain-ollama", "pandas"):
        try:
            resolved[package] = version(package)
        except PackageNotFoundError:
            resolved[package] = "not-installed"
    return resolved


def _create_runtime(arguments: argparse.Namespace):
    vector_store = create_vector_store(
        arguments.data,
        database_path=arguments.database,
        embedding_model=arguments.embedding_model,
        ollama_host=arguments.ollama_host,
    )
    model = create_chat_model(
        model=arguments.chat_model,
        base_url=arguments.ollama_host,
    )
    return vector_store, model


def _answer(arguments: argparse.Namespace, question: str) -> None:
    vector_store, model = _create_runtime(arguments)
    result = answer_question(
        question,
        vector_store=vector_store,
        model=model,
        limit=arguments.limit,
        min_rating=arguments.min_rating,
        max_rating=arguments.max_rating,
        start_date=arguments.start_date,
        end_date=arguments.end_date,
        sentiments=arguments.sentiment,
        restaurants=arguments.restaurant,
        countries=arguments.country,
    )
    _print_answer(result)


def run(arguments: argparse.Namespace) -> int:
    dataframe = load_reviews(arguments.data)
    health = _health_for_arguments(arguments)

    if arguments.command == "status":
        print(f"Dataset: {arguments.data}")
        print(f"Reviews: {len(dataframe)}")
        _print_health(health)
        return 0 if health.ok else 1

    if not health.ok:
        _print_health(health)
        return 1

    if arguments.command == "evaluate":
        vector_store, model = _create_runtime(arguments)
        cases = load_evaluation_cases(
            arguments.cases,
            dataset_path=arguments.data,
        )
        metrics, observations = run_rag_evaluation(
            cases,
            vector_store=vector_store,
            model=model,
            limit=arguments.limit,
        )
        semantic_metrics = retrieval_metrics_from_observations(
            cases,
            observations,
            limit=arguments.limit,
        )
        baseline_metrics, baseline_observations = run_bm25_baseline(
            cases,
            vector_store=vector_store,
            limit=arguments.limit,
        )
        print(json.dumps(metrics.as_dict(), indent=2, sort_keys=True))
        for observation in observations:
            print(
                f"{observation.case_id}: retrieved={len(observation.retrieved_source_ids)} "
                f"cited={len(observation.cited_source_ids)} "
                f"abstained={observation.abstained}"
            )
        if arguments.report_dir is not None:
            report = build_evaluation_report(
                cases=cases,
                rag_metrics=metrics,
                semantic_metrics=semantic_metrics,
                baseline_metrics=baseline_metrics,
                observations=observations,
                baseline_observations=baseline_observations,
                configuration={
                    "chat_model": arguments.chat_model,
                    "embedding_model": arguments.embedding_model,
                    "ollama_version": ollama_version(arguments.ollama_host),
                    "ollama_host": _safe_endpoint(arguments.ollama_host),
                    "evidence_limit": arguments.limit,
                    "models": model_metadata(
                        (arguments.chat_model, arguments.embedding_model),
                        host=arguments.ollama_host,
                    ),
                },
                provenance={
                    "application_version": __version__,
                    "python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "dataset_file": arguments.data.name,
                    "dataset_sha256": _file_sha256(arguments.data),
                    "review_count": len(dataframe),
                    "cases_file": arguments.cases.name,
                    "cases_sha256": _file_sha256(arguments.cases),
                    "dependency_versions": _dependency_versions(),
                    **_git_provenance(),
                },
            )
            json_path = arguments.report_dir / "evaluation-report.json"
            markdown_path = arguments.report_dir / "README.md"
            write_evaluation_report(
                report,
                json_path=json_path,
                markdown_path=markdown_path,
            )
            print(f"Wrote {json_path}")
            print(f"Wrote {markdown_path}")
        return 0

    if arguments.command == "ask":
        _answer(arguments, arguments.question)
        return 0

    print("Local Restaurant Review Intelligence")
    print("Type q, quit, or exit to stop.")
    while True:
        try:
            question = input("\nAsk a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"q", "quit", "exit"}:
            return 0
        if not question:
            continue
        _answer(arguments, question)


def main(argv: Sequence[str] | None = None) -> int:
    supplied_arguments = list(argv) if argv is not None else sys.argv[1:]
    if not supplied_arguments:
        supplied_arguments = ["chat"]
    parser = build_parser()
    arguments = parser.parse_args(supplied_arguments)
    try:
        return run(arguments)
    except (ReviewDataError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

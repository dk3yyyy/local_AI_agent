import argparse
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from agent import answer_question, create_chat_model
from ollama_health import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_HOST,
    check_ollama,
)
from vector import (
    DEFAULT_DATA_PATH,
    DEFAULT_DATABASE_PATH,
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
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)


def _add_search_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--min-rating", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--max-rating", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--start-date", type=_date_argument)
    parser.add_argument("--end-date", type=_date_argument)


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
        print(
            f"[{source.citation_number}] "
            f"{metadata.get('rating', '?')}/5, {metadata.get('date', 'unknown')}"
        )
        print(source.document.page_content.replace("\n", " — "))


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
    except (ReviewDataError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

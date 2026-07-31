import os
import runpy
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pandas as pd
from langchain_chroma import Chroma as RealChroma

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTOR_MODULE = PROJECT_ROOT / "vector.py"


class FakeEmbeddings:
    document_batches: ClassVar[list[list[str]]] = []

    def __init__(self, *, model: str) -> None:
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_batches.append(texts)
        return [[float(index), 0.0, 1.0] for index, _ in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [0.0, 0.0, 1.0]


class FailingEmbeddings(FakeEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding failed")


class FakeChroma:
    existing_ids: ClassVar[list[str]] = []
    add_calls: ClassVar[list[dict[str, object]]] = []

    def __init__(
        self,
        *,
        collection_name: str,
        persist_directory: str,
        embedding_function: FakeEmbeddings,
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function

    def get(self, *, ids: list[str], include: list[str]) -> dict[str, list[str]]:
        return {"ids": [item_id for item_id in ids if item_id in self.existing_ids]}

    def add_documents(self, *, documents: list[object], ids: list[str]) -> None:
        self.add_calls.append({"documents": documents, "ids": ids})

    def as_retriever(self, *, search_kwargs: dict[str, int]) -> dict[str, object]:
        return {"store": self, "search_kwargs": search_kwargs}


class VectorStoreInitializationTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeEmbeddings.document_batches = []
        FakeChroma.existing_ids = []
        FakeChroma.add_calls = []
        self.dataframe = pd.DataFrame(
            [
                {
                    "Title": "Great crust",
                    "Date": "2026-01-01",
                    "Rating": 5,
                    "Review": "Crisp and flavorful.",
                },
                {
                    "Title": "Slow service",
                    "Date": "2026-01-02",
                    "Rating": 2,
                    "Review": "The food arrived cold.",
                },
            ]
        )

    def run_vector_module(self, working_directory: Path) -> None:
        previous_directory = Path.cwd()
        os.chdir(working_directory)
        try:
            with (
                patch("pandas.read_csv", return_value=self.dataframe) as read_csv,
                patch("langchain_ollama.OllamaEmbeddings", FakeEmbeddings),
                patch("langchain_chroma.Chroma", FakeChroma),
            ):
                runpy.run_path(str(VECTOR_MODULE), run_name="vector_under_test")
                self.read_csv_argument = read_csv.call_args.args[0]
        finally:
            os.chdir(previous_directory)

    def test_empty_existing_database_is_populated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            working_directory = Path(temporary_directory)
            (working_directory / "chrome_langchain_db").mkdir()

            self.run_vector_module(working_directory)

        self.assertEqual(len(FakeChroma.add_calls), 1)
        self.assertEqual(FakeChroma.add_calls[0]["ids"], ["0", "1"])

    def test_only_missing_documents_are_added(self) -> None:
        FakeChroma.existing_ids = ["0"]

        with tempfile.TemporaryDirectory() as temporary_directory:
            working_directory = Path(temporary_directory)
            (working_directory / "chrome_langchain_db").mkdir()

            self.run_vector_module(working_directory)

        self.assertEqual(len(FakeChroma.add_calls), 1)
        self.assertEqual(FakeChroma.add_calls[0]["ids"], ["1"])

    def test_recovers_after_real_chroma_initialization_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            working_directory = Path(temporary_directory)
            database_directory = working_directory / "chrome_langchain_db"
            stores: list[RealChroma] = []

            def create_chroma(**kwargs: object) -> RealChroma:
                kwargs["persist_directory"] = str(database_directory)
                store = RealChroma(**kwargs)
                stores.append(store)
                return store

            previous_directory = Path.cwd()
            os.chdir(working_directory)
            try:
                with (
                    patch("pandas.read_csv", return_value=self.dataframe),
                    patch("langchain_ollama.OllamaEmbeddings", FailingEmbeddings),
                    patch("langchain_chroma.Chroma", side_effect=create_chroma),
                    self.assertRaisesRegex(RuntimeError, "embedding failed"),
                ):
                    runpy.run_path(
                        str(VECTOR_MODULE), run_name="first_failed_initialization"
                    )

                self.assertTrue(database_directory.exists())

                with (
                    patch("pandas.read_csv", return_value=self.dataframe),
                    patch("langchain_ollama.OllamaEmbeddings", FakeEmbeddings),
                    patch("langchain_chroma.Chroma", side_effect=create_chroma),
                ):
                    runpy.run_path(
                        str(VECTOR_MODULE), run_name="recovered_initialization"
                    )
            finally:
                os.chdir(previous_directory)

            self.assertEqual(stores[-1]._collection.count(), 2)
            self.assertEqual(len(FakeEmbeddings.document_batches), 1)

    def test_dataset_path_is_independent_of_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            self.run_vector_module(Path(temporary_directory))

        self.assertEqual(
            Path(self.read_csv_argument),
            PROJECT_ROOT / "realistic_restaurant_reviews.csv",
        )


if __name__ == "__main__":
    unittest.main()

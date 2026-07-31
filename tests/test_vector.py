import os
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from vector import (
    DEFAULT_DATA_PATH,
    ReviewDataError,
    _documents_and_ids,
    create_vector_store,
    dataset_fingerprint,
    dataset_storage,
    dataset_summary,
    index_count,
    load_reviews,
    search_reviews,
)


class DeterministicEmbeddings:
    def __init__(self) -> None:
        self.document_batches: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_batches.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        total = sum(text.encode("utf-8"))
        return [float((total + offset) % 17) for offset in range(8)]


class FailingEmbeddings(DeterministicEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("simulated embedding failure")


class FakeVectorStore:
    def __init__(self, results: list[tuple[Document, float]]) -> None:
        self.results = results
        self.requested_k: int | None = None

    def get(self, *, include: list[str]) -> dict[str, list[str]]:
        return {"ids": [str(index) for index in range(len(self.results))]}

    def similarity_search_with_score(
        self, query: str, *, k: int
    ) -> list[tuple[Document, float]]:
        self.requested_k = k
        return self.results[:k]


class ReviewDataTest(unittest.TestCase):
    def test_default_dataset_path_is_independent_of_working_directory(self) -> None:
        previous_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            try:
                os.chdir(temporary_directory)
                dataframe = load_reviews()
            finally:
                os.chdir(previous_directory)

        self.assertEqual(len(dataframe), 123)
        self.assertTrue(DEFAULT_DATA_PATH.is_file())
        self.assertEqual(DEFAULT_DATA_PATH.name, "realistic_restaurant_reviews.csv")
        self.assertEqual(DEFAULT_DATA_PATH.parent.name, "data")

    def test_rejects_missing_columns(self) -> None:
        dataframe = pd.DataFrame({"Title": ["Only a title"]})

        with self.assertRaisesRegex(ReviewDataError, "review text column"):
            load_reviews(dataframe)

    def test_rejects_invalid_rating_and_date(self) -> None:
        dataframe = pd.DataFrame(
            {
                "Title": ["Bad row"],
                "Date": ["not-a-date"],
                "Rating": [7],
                "Review": ["Invalid values"],
            }
        )

        with self.assertRaisesRegex(ReviewDataError, "Rating"):
            load_reviews(dataframe)

    def test_dataset_summary_is_deterministic(self) -> None:
        dataframe = load_reviews(
            pd.DataFrame(
                {
                    "Title": ["A", "B", "C"],
                    "Date": ["2024-01-01", "2024-02-01", "2024-03-01"],
                    "Rating": [1, 4, 5],
                    "Review": ["Bad", "Good", "Great"],
                }
            )
        )

        summary = dataset_summary(dataframe)

        self.assertEqual(summary.total_reviews, 3)
        self.assertAlmostEqual(summary.average_rating, 10 / 3)
        self.assertEqual(summary.high_rated, 2)
        self.assertEqual(summary.low_rated, 1)
        self.assertEqual(summary.rating_counts, {1: 1, 2: 0, 3: 0, 4: 1, 5: 1})


class VectorStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataframe = pd.DataFrame(
            {
                "Title": ["Great crust", "Slow service"],
                "Date": ["2024-01-10", "2024-02-20"],
                "Rating": [5, 2],
                "Review": ["Crisp and flavorful.", "Our order arrived late."],
            }
        )

    def test_recovers_after_failed_real_chroma_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "chroma"

            with self.assertRaisesRegex(RuntimeError, "simulated embedding failure"):
                create_vector_store(
                    self.dataframe,
                    database_path=database_path,
                    collection_name="recovery_test",
                    embeddings=FailingEmbeddings(),
                )

            self.assertTrue(database_path.exists())

            embeddings = DeterministicEmbeddings()
            store = create_vector_store(
                self.dataframe,
                database_path=database_path,
                collection_name="recovery_test",
                embeddings=embeddings,
            )
            self.assertEqual(index_count(store), 2)
            self.assertEqual(len(embeddings.document_batches), 1)

            second_embeddings = DeterministicEmbeddings()
            second_store = create_vector_store(
                self.dataframe,
                database_path=database_path,
                collection_name="recovery_test",
                embeddings=second_embeddings,
            )
            self.assertEqual(index_count(second_store), 2)
            self.assertEqual(second_embeddings.document_batches, [])

    def test_adds_only_rows_missing_from_partial_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "chroma"
            first_embeddings = DeterministicEmbeddings()
            create_vector_store(
                self.dataframe.iloc[:1],
                database_path=database_path,
                collection_name="partial_test",
                embeddings=first_embeddings,
            )

            second_embeddings = DeterministicEmbeddings()
            store = create_vector_store(
                self.dataframe,
                database_path=database_path,
                collection_name="partial_test",
                embeddings=second_embeddings,
            )

            self.assertEqual(index_count(store), 2)
            self.assertEqual(len(second_embeddings.document_batches), 1)
            self.assertEqual(len(second_embeddings.document_batches[0]), 1)

    def test_document_ids_are_stable_when_rows_are_reordered(self) -> None:
        normalized = load_reviews(self.dataframe)
        _, original_ids = _documents_and_ids(normalized)
        _, reordered_ids = _documents_and_ids(
            normalized.iloc[::-1].reset_index(drop=True)
        )

        self.assertEqual(set(original_ids), set(reordered_ids))
        self.assertTrue(all(item_id.startswith("review_") for item_id in original_ids))

    def test_reconciles_changed_and_deleted_reviews(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "chroma"
            first = create_vector_store(
                self.dataframe,
                database_path=database_path,
                collection_name="reconciliation_test",
                embeddings=DeterministicEmbeddings(),
            )
            first_ids = set(first.get(include=[])["ids"])

            replacement = self.dataframe.iloc[[0]].copy()
            replacement.loc[replacement.index[0], "Review"] = "Updated crisp crust."
            second = create_vector_store(
                replacement,
                database_path=database_path,
                collection_name="reconciliation_test",
                embeddings=DeterministicEmbeddings(),
            )
            second_ids = set(second.get(include=[])["ids"])

            self.assertEqual(index_count(second), 1)
            self.assertTrue(first_ids.isdisjoint(second_ids))

    def test_dataset_storage_is_derived_and_isolated(self) -> None:
        normalized = load_reviews(self.dataframe)
        changed = normalized.copy()
        changed.loc[0, "Review"] = "Different source content"

        first_digest = dataset_fingerprint(normalized)
        second_digest = dataset_fingerprint(changed)
        first_database, first_collection = dataset_storage(normalized)
        second_database, second_collection = dataset_storage(changed)

        self.assertNotEqual(first_digest, second_digest)
        self.assertNotEqual(first_database, second_database)
        self.assertNotEqual(first_collection, second_collection)
        self.assertIn(first_digest, str(first_database))
        self.assertTrue(first_collection.startswith("reviews_"))

    def test_filters_semantic_results_by_rating_and_date(self) -> None:
        results = [
            (
                Document(
                    page_content="Excellent pizza",
                    metadata={"rating": 5, "date": "2024-03-10"},
                    id="a",
                ),
                0.1,
            ),
            (
                Document(
                    page_content="Old positive review",
                    metadata={"rating": 4, "date": "2024-01-01"},
                    id="b",
                ),
                0.2,
            ),
            (
                Document(
                    page_content="Recent complaint",
                    metadata={"rating": 2, "date": "2024-03-15"},
                    id="c",
                ),
                0.3,
            ),
        ]
        store = FakeVectorStore(results)

        matches = search_reviews(
            store,
            "pizza",
            limit=5,
            min_rating=4,
            max_rating=5,
            start_date=date(2024, 2, 1),
            end_date=date(2024, 3, 31),
        )

        self.assertEqual([match.document.id for match in matches], ["a"])
        self.assertEqual(store.requested_k, 3)


if __name__ == "__main__":
    unittest.main()

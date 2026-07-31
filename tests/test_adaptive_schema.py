import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from dashboard_support import prepare_uploaded_dataset
from vector import (
    ColumnMapping,
    ReviewDataError,
    create_vector_store,
    dataset_summary,
    detect_column_mapping,
    filter_reviews,
    load_reviews,
    search_reviews,
    suggest_column_mapping,
)


class DeterministicEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [float((sum(text.encode()) + offset) % 17) for offset in range(8)]
            for text in texts
        ]

    def embed_query(self, text: str) -> list[float]:
        return [float((sum(text.encode()) + offset) % 17) for offset in range(8)]


class FakeVectorStore:
    def __init__(self, results: list[tuple[Document, float]]) -> None:
        self.results = results

    def get(self, *, include: list[str]) -> dict[str, list[str]]:
        return {"ids": [str(index) for index in range(len(self.results))]}

    def similarity_search_with_score(
        self, query: str, *, k: int
    ) -> list[tuple[Document, float]]:
        return self.results[:k]


class AdaptiveSchemaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.platform_dataframe = pd.DataFrame(
            {
                "Country": ["Nigeria", "United Kingdom", "Nigeria"],
                "Restaurant Name": ["Atlas Pizza", "North Pie", "Atlas Pizza"],
                "Sentiment": ["Positive", "Negative", "Neutral"],
                "Review Title": ["Great crust", "Slow service", "Fair meal"],
                "Review Date": ["2026-01-01", "2026-01-02", "2026-01-03"],
                "Review": [
                    "The crust was crisp.",
                    "Service took too long.",
                    "The meal was acceptable.",
                ],
            }
        )

    def test_detects_aliases_without_a_rating_column(self) -> None:
        mapping = detect_column_mapping(self.platform_dataframe.columns)

        self.assertEqual(mapping.review, "Review")
        self.assertEqual(mapping.title, "Review Title")
        self.assertEqual(mapping.date, "Review Date")
        self.assertIsNone(mapping.rating)
        self.assertEqual(mapping.sentiment, "Sentiment")
        self.assertEqual(mapping.restaurant, "Restaurant Name")
        self.assertEqual(mapping.country, "Country")

        reviews = load_reviews(self.platform_dataframe)
        summary = dataset_summary(reviews)

        self.assertEqual(len(reviews), 3)
        self.assertTrue(reviews["Rating"].isna().all())
        self.assertEqual(summary.average_rating, None)
        self.assertEqual(
            summary.sentiment_counts, {"Negative": 1, "Neutral": 1, "Positive": 1}
        )
        self.assertEqual(summary.first_date, date(2026, 1, 1))
        self.assertEqual(summary.restaurant_count, 2)
        self.assertEqual(summary.country_count, 2)

    def test_manual_mapping_supports_unknown_column_names(self) -> None:
        dataframe = pd.DataFrame(
            {
                "Venue": ["Cafe A"],
                "Published On": ["2026-02-01"],
                "Customer Words": ["A calm room and excellent coffee."],
                "Mood": ["Happy"],
            }
        )
        mapping = ColumnMapping(
            review="Customer Words",
            date="Published On",
            sentiment="Mood",
            restaurant="Venue",
        )

        reviews = load_reviews(dataframe, mapping=mapping)

        self.assertEqual(reviews.loc[0, "Review"], "A calm room and excellent coffee.")
        self.assertEqual(reviews.loc[0, "Date"], "2026-02-01")
        self.assertEqual(reviews.loc[0, "Sentiment"], "Happy")
        self.assertEqual(reviews.loc[0, "Restaurant"], "Cafe A")
        self.assertTrue(pd.isna(reviews.loc[0, "Title"]))

    def test_requires_a_review_mapping_but_not_other_fields(self) -> None:
        suggestions = suggest_column_mapping(["Identifier", "Notes Code"])
        self.assertIsNone(suggestions["review"])

        with self.assertRaisesRegex(ReviewDataError, "review text column"):
            detect_column_mapping(["Identifier", "Notes Code"])

        reviews = load_reviews(
            pd.DataFrame({"Comment": ["Only review text is required."]})
        )
        self.assertEqual(reviews.loc[0, "Review"], "Only review text is required.")
        self.assertTrue(reviews["Rating"].isna().all())
        self.assertTrue(reviews["Date"].isna().all())

    def test_rejects_duplicate_role_assignments(self) -> None:
        mapping = ColumnMapping(review="Text", title="Text")

        with self.assertRaisesRegex(ReviewDataError, "more than one role"):
            load_reviews(pd.DataFrame({"Text": ["Hello"]}), mapping=mapping)

    def test_adaptive_filters_use_available_metadata(self) -> None:
        reviews = load_reviews(self.platform_dataframe)

        filtered = filter_reviews(
            reviews,
            sentiments=("Positive",),
            restaurants=("Atlas Pizza",),
            countries=("Nigeria",),
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 2),
        )

        self.assertEqual(filtered["Review"].tolist(), ["The crust was crisp."])

    def test_search_accepts_records_without_rating_or_date(self) -> None:
        store = FakeVectorStore(
            [
                (
                    Document(
                        page_content="Great coffee",
                        metadata={
                            "sentiment": "Positive",
                            "restaurant": "Cafe A",
                            "country": "Ghana",
                        },
                        id="a",
                    ),
                    0.1,
                ),
                (
                    Document(
                        page_content="Slow service",
                        metadata={"sentiment": "Negative", "restaurant": "Cafe B"},
                        id="b",
                    ),
                    0.2,
                ),
            ]
        )

        matches = search_reviews(
            store,
            "coffee",
            sentiments=("Positive",),
            countries=("Ghana",),
        )

        self.assertEqual([match.document.id for match in matches], ["a"])

    def test_vector_metadata_retains_adaptive_fields_and_extras(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = create_vector_store(
                self.platform_dataframe,
                database_path=Path(temporary_directory) / "chroma",
                collection_name="adaptive_schema",
                embeddings=DeterministicEmbeddings(),
            )
            result = store.get(ids=["0"], include=["metadatas", "documents"])

        metadata = result["metadatas"][0]
        self.assertEqual(metadata["sentiment"], "Positive")
        self.assertEqual(metadata["restaurant"], "Atlas Pizza")
        self.assertEqual(metadata["country"], "Nigeria")
        self.assertIn("Great crust", result["documents"][0])

    def test_upload_storage_changes_when_mapping_changes(self) -> None:
        content = b"First,Second\nTitle text,Review text\n"
        review_first = ColumnMapping(review="First", title="Second")
        review_second = ColumnMapping(review="Second", title="First")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = prepare_uploaded_dataset(
                content,
                mapping=review_first,
                storage_root=root,
            )
            second = prepare_uploaded_dataset(
                content,
                mapping=review_second,
                storage_root=root,
            )

        self.assertNotEqual(first.digest, second.digest)
        self.assertNotEqual(first.database_path, second.database_path)
        self.assertEqual(first.mapping, review_first)
        self.assertEqual(second.mapping, review_second)

    def test_weak_alias_substrings_remain_unmapped(self) -> None:
        suggestions = suggest_column_mapping(["Review", "Label ID", "Region Code"])

        self.assertEqual(suggestions["review"], "Review")
        self.assertIsNone(suggestions["sentiment"])
        self.assertIsNone(suggestions["country"])

    def test_date_filter_excludes_undated_rows(self) -> None:
        reviews = load_reviews(
            pd.DataFrame(
                {
                    "Review": ["Dated", "Undated"],
                    "Review Date": ["2026-01-01", None],
                }
            )
        )

        filtered = filter_reviews(
            reviews,
            start_date=date(2025, 12, 31),
            end_date=date(2026, 1, 2),
        )

        self.assertEqual(filtered["Review"].tolist(), ["Dated"])

    def test_extra_column_name_collisions_preserve_both_values(self) -> None:
        dataframe = pd.DataFrame(
            [["Good meal", "first", "second"]],
            columns=["Review", "Order ID", "order_id"],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = create_vector_store(
                dataframe,
                database_path=Path(temporary_directory) / "chroma",
                collection_name="extra_collision",
                embeddings=DeterministicEmbeddings(),
            )
            result = store.get(ids=["0"], include=["metadatas", "documents"])

        metadata = result["metadatas"][0]
        self.assertEqual(metadata["extra_orderid"], "first")
        self.assertEqual(metadata["extra_orderid_2"], "second")
        self.assertIn("Order ID: first", result["documents"][0])
        self.assertIn("order_id: second", result["documents"][0])


if __name__ == "__main__":
    unittest.main()

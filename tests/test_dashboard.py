import os
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest

import agent
import ollama_health
import vector


class MixedTokenModel:
    def invoke(self, _: str) -> str:
        return "Guests praise the crispy crust [1].\n\nINSUFFICIENT_EVIDENCE"


class DashboardStore:
    def __init__(self) -> None:
        self.document = Document(
            page_content="The crust was perfectly crispy.",
            metadata={
                "source_id": "review-1",
                "rating": 5,
                "date": "2024-01-10",
            },
            id="review-1",
        )

    def get(self, **_: object) -> dict[str, list[str]]:
        return {"ids": ["review-1"]}

    def similarity_search_with_score(
        self, _: str, *, k: int, filter: dict | None = None
    ) -> list[tuple[Document, float]]:
        del filter
        return [(self.document, 0.1)][:k]


class DashboardRenderTest(unittest.TestCase):
    def test_renders_analytics_and_safe_offline_state(self) -> None:
        with patch.object(
            ollama_health,
            "DEFAULT_OLLAMA_HOST",
            "http://127.0.0.1:9",
        ):
            application = AppTest.from_file("dashboard.py").run(timeout=30)

        self.assertEqual(list(application.exception), [])
        self.assertEqual(
            [metric.value for metric in application.metric],
            ["123", "3.59", "63", "40"],
        )
        self.assertEqual(len(application.chat_input), 1)
        self.assertTrue(application.chat_input[0].disabled)
        self.assertIn(
            "Chat is disabled until Ollama is running",
            application.warning[0].value,
        )

    def test_adapts_uploaded_platform_schema(self) -> None:
        content = b"""Country,Restaurant Name,Sentiment,Review Title,Review Date,Review
Nigeria,Atlas Pizza,Positive,Great crust,2026-01-01,The crust was crisp.
United Kingdom,North Pie,Negative,Slow service,2026-01-02,Service took too long.
Nigeria,Atlas Pizza,Neutral,Fair meal,2026-01-03,The meal was acceptable.
"""
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            patch.object(
                ollama_health,
                "DEFAULT_OLLAMA_HOST",
                "http://127.0.0.1:9",
            ),
            patch.dict(
                os.environ,
                {"LOCAL_AI_STORAGE_ROOT": temporary_directory},
            ),
        ):
            application = AppTest.from_file("dashboard.py").run(timeout=30)
            application.file_uploader[0].set_value(
                ("platform-reviews.csv", content, "text/csv")
            ).run(timeout=30)

        self.assertEqual(list(application.exception), [])
        mappings = {item.label: item.value for item in application.selectbox}
        self.assertEqual(mappings["Review text"], "Review")
        self.assertEqual(mappings["Review title"], "Review Title")
        self.assertEqual(mappings["Review date"], "Review Date")
        self.assertEqual(mappings["Rating or stars"], "Not mapped")
        self.assertEqual(mappings["Sentiment"], "Sentiment")
        self.assertEqual(mappings["Restaurant"], "Restaurant Name")
        self.assertEqual(mappings["Country or region"], "Country")
        self.assertEqual(
            [metric.value for metric in application.metric],
            ["3", "3", "1", "1"],
        )
        self.assertIn(
            "Sentiment distribution",
            [heading.value for heading in application.subheader],
        )
        self.assertEqual(
            [item.label for item in application.multiselect],
            ["Sentiment", "Restaurant", "Country or region"],
        )

    def test_does_not_render_mixed_insufficient_evidence_token(self) -> None:
        health = ollama_health.OllamaHealth(
            True,
            ("llama3.2:latest", "mxbai-embed-large:latest"),
            (),
        )
        host = "http://dashboard-token-regression:11434"
        with (
            patch.object(ollama_health, "DEFAULT_OLLAMA_HOST", host),
            patch.object(ollama_health, "check_ollama", return_value=health),
            patch.object(vector, "create_vector_store", return_value=DashboardStore()),
            patch.object(agent, "create_chat_model", return_value=MixedTokenModel()),
        ):
            application = AppTest.from_file("dashboard.py").run(timeout=30)
            application.chat_input[0].set_value(
                "What do guests say about the crust?"
            ).run(timeout=30)

        self.assertEqual(list(application.exception), [])
        rendered_markdown = "\n".join(item.value for item in application.markdown)
        self.assertIn("I could not produce an answer with citations", rendered_markdown)
        self.assertNotIn("INSUFFICIENT_EVIDENCE", rendered_markdown)
        self.assertNotIn("#### Evidence", rendered_markdown)


if __name__ == "__main__":
    unittest.main()

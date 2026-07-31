import os
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import ollama_health


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


if __name__ == "__main__":
    unittest.main()

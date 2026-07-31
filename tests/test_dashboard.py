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


if __name__ == "__main__":
    unittest.main()

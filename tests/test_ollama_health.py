import unittest

from ollama_health import check_ollama


class FakeClient:
    def __init__(
        self, models: list[str] | None = None, error: Exception | None = None
    ) -> None:
        self.models = models or []
        self.error = error

    def list(self) -> dict[str, list[dict[str, str]]]:
        if self.error is not None:
            raise self.error
        return {"models": [{"model": model} for model in self.models]}


class OllamaHealthTest(unittest.TestCase):
    def test_reports_available_models(self) -> None:
        health = check_ollama(
            required_models=("llama3.2", "mxbai-embed-large"),
            client=FakeClient(["llama3.2:latest", "mxbai-embed-large:latest"]),
        )

        self.assertTrue(health.ok)
        self.assertEqual(health.missing_models, ())
        self.assertEqual(
            health.available_models,
            ("llama3.2:latest", "mxbai-embed-large:latest"),
        )

    def test_reports_missing_models(self) -> None:
        health = check_ollama(
            required_models=("llama3.2", "mxbai-embed-large"),
            client=FakeClient(["llama3.2:latest"]),
        )

        self.assertFalse(health.ok)
        self.assertTrue(health.service_available)
        self.assertEqual(health.missing_models, ("mxbai-embed-large",))
        self.assertIn("ollama pull mxbai-embed-large", health.instructions)

    def test_reports_unavailable_service(self) -> None:
        health = check_ollama(
            required_models=("llama3.2",),
            client=FakeClient(error=ConnectionError("connection refused")),
        )

        self.assertFalse(health.ok)
        self.assertFalse(health.service_available)
        self.assertIn("ollama serve", health.instructions)
        self.assertIn("connection refused", health.error or "")


if __name__ == "__main__":
    unittest.main()

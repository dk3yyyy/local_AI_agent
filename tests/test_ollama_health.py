import unittest

from ollama_health import check_ollama, model_metadata, ollama_version


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
    def test_reports_server_version(self) -> None:
        class Response:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, str]:
                return {"version": "0.32.5"}

        def request(url: str, *, timeout: float):
            self.assertEqual(url, "http://localhost:11434/api/version")
            self.assertEqual(timeout, 5.0)
            return Response()

        self.assertEqual(
            ollama_version("http://localhost:11434/", request=request),
            "0.32.5",
        )

    def test_model_metadata_resolves_tags_and_preserves_digests(self) -> None:
        class MetadataClient:
            def list(self):
                return {
                    "models": [
                        {
                            "model": "llama3.2:latest",
                            "digest": "sha256:chat",
                            "size": 123,
                        },
                        {
                            "model": "mxbai-embed-large:latest",
                            "digest": "sha256:embed",
                            "size": 456,
                        },
                    ]
                }

        metadata = model_metadata(
            ("llama3.2", "mxbai-embed-large"),
            client=MetadataClient(),
        )

        self.assertEqual(metadata["llama3.2"]["resolved_name"], "llama3.2:latest")
        self.assertEqual(metadata["llama3.2"]["digest"], "sha256:chat")
        self.assertEqual(metadata["mxbai-embed-large"]["size"], 456)

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

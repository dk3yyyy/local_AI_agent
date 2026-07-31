import os
from dataclasses import dataclass
from typing import Any

from httpx import HTTPError
from ollama import Client, ResponseError

DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")


@dataclass(frozen=True)
class OllamaHealth:
    service_available: bool
    available_models: tuple[str, ...]
    missing_models: tuple[str, ...]
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.service_available and not self.missing_models

    @property
    def instructions(self) -> str:
        if not self.service_available:
            return "Start Ollama with `ollama serve`, then try again."
        if self.missing_models:
            commands = "\n".join(
                f"ollama pull {model}" for model in self.missing_models
            )
            return f"Download the missing models:\n{commands}"
        return "Ollama is ready."


def create_ollama_client(host: str = DEFAULT_OLLAMA_HOST) -> Client:
    return Client(host=host)


def _available_model_names(response: Any) -> tuple[str, ...]:
    models = (
        response.get("models", []) if isinstance(response, dict) else response.models
    )
    names: list[str] = []
    for model in models:
        if isinstance(model, dict):
            name = model.get("model") or model.get("name")
        else:
            name = getattr(model, "model", None) or getattr(model, "name", None)
        if name:
            names.append(str(name))
    return tuple(names)


def _model_is_available(required: str, available: tuple[str, ...]) -> bool:
    if ":" in required:
        return required in available
    return any(
        name == required or name.startswith(f"{required}:") for name in available
    )


def check_ollama(
    *,
    required_models: tuple[str, ...] = (
        DEFAULT_CHAT_MODEL,
        DEFAULT_EMBEDDING_MODEL,
    ),
    host: str = DEFAULT_OLLAMA_HOST,
    client: Any | None = None,
) -> OllamaHealth:
    """Check service reachability and required model availability."""
    ollama_client = client or create_ollama_client(host)
    try:
        available_models = _available_model_names(ollama_client.list())
    except (ConnectionError, HTTPError, OSError, ResponseError) as error:
        return OllamaHealth(
            service_available=False,
            available_models=(),
            missing_models=required_models,
            error=str(error),
        )

    missing_models = tuple(
        model
        for model in required_models
        if not _model_is_available(model, available_models)
    )
    return OllamaHealth(
        service_available=True,
        available_models=available_models,
        missing_models=missing_models,
    )

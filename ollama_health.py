import os
from dataclasses import dataclass
from typing import Any

import httpx
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


def ollama_version(
    host: str = DEFAULT_OLLAMA_HOST,
    *,
    request: Any | None = None,
) -> str:
    """Return the server-reported Ollama runtime version."""
    getter = request or httpx.get
    response = getter(f"{host.rstrip('/')}/api/version", timeout=5.0)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not str(payload.get("version") or "").strip():
        raise ValueError("Ollama version response is missing a version")
    return str(payload["version"])


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


def _model_value(model: Any, key: str) -> Any:
    if isinstance(model, dict):
        return model.get(key)
    return getattr(model, key, None)


def model_metadata(
    required_models: tuple[str, ...],
    *,
    host: str = DEFAULT_OLLAMA_HOST,
    client: Any | None = None,
) -> dict[str, dict[str, str | int | None]]:
    """Resolve configured model tags to the immutable digests Ollama reports."""
    ollama_client = client or create_ollama_client(host)
    response = ollama_client.list()
    models = (
        response.get("models", []) if isinstance(response, dict) else response.models
    )
    resolved: dict[str, dict[str, str | int | None]] = {}
    for required in required_models:
        candidates = []
        for model in models:
            name = _model_value(model, "model") or _model_value(model, "name")
            if name and _model_is_available(required, (str(name),)):
                candidates.append(model)
        if not candidates:
            continue
        selected = min(
            candidates,
            key=lambda model: (
                str(_model_value(model, "model") or _model_value(model, "name"))
                != required,
                not str(
                    _model_value(model, "model") or _model_value(model, "name")
                ).endswith(":latest"),
                str(_model_value(model, "model") or _model_value(model, "name")),
            ),
        )
        size = _model_value(selected, "size")
        resolved[required] = {
            "resolved_name": str(
                _model_value(selected, "model") or _model_value(selected, "name")
            ),
            "digest": str(_model_value(selected, "digest") or "") or None,
            "size": int(size) if size is not None else None,
            "modified_at": str(_model_value(selected, "modified_at") or "") or None,
        }
    return resolved


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

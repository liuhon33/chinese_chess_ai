from __future__ import annotations

import os

from cchess_alphazero.agent.backends.base import ModelBackend


def get_backend_name(config=None) -> str:
    configured = getattr(getattr(config, "opts", None), "backend", None)
    return (configured or os.environ.get("CCHESS_BACKEND") or "torch").lower()


def create_model_backend(config) -> ModelBackend:
    backend_name = get_backend_name(config)
    if backend_name == "keras":
        from cchess_alphazero.agent.backends.keras_backend import KerasModelBackend

        return KerasModelBackend(config)
    if backend_name in {"torch", "pytorch"}:
        from cchess_alphazero.agent.backends.torch_backend import TorchModelBackend

        return TorchModelBackend(config)
    raise ValueError(f"Unsupported model backend: {backend_name}")


def configure_backend_session(config=None, **kwargs):
    backend_name = get_backend_name(config)
    if backend_name == "keras":
        from cchess_alphazero.agent.backends.keras_backend import configure_keras_session

        return configure_keras_session(**kwargs)
    if backend_name in {"torch", "pytorch"}:
        from cchess_alphazero.agent.backends.torch_backend import configure_torch_session

        return configure_torch_session(**kwargs)
    raise ValueError(f"Unsupported model backend: {backend_name}")

from __future__ import annotations

import platform
from typing import Any

from app.models.base import BaseModelAdapter


def detect_device(override: str = "auto") -> str:
    if override != "auto":
        return override
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class ModelManager:
    def __init__(self, device: str, unload_after_request: bool = True):
        self.device = device
        self.unload_after_request = unload_after_request
        self._adapters: dict[str, BaseModelAdapter] = {}

    def register(self, model_id: str, adapter: BaseModelAdapter) -> None:
        self._adapters[model_id] = adapter

    def run(self, model_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
        adapter = self._adapters[model_id]
        adapter.load()
        try:
            return adapter.predict(inputs)
        finally:
            if self.unload_after_request:
                adapter.unload()
                self.clear_memory()

    def clear_memory(self) -> None:
        try:
            import torch

            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps" and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except ImportError:
            return

    @staticmethod
    def runtime_info() -> dict[str, str]:
        try:
            import torch

            version = torch.__version__
        except ImportError:
            version = "not installed"
        return {"python": platform.python_version(), "pytorch": version}

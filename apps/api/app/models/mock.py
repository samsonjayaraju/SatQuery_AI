from __future__ import annotations

from typing import Any

from app.models.base import BaseModelAdapter


class MockModelAdapter(BaseModelAdapter):
    """Deterministic development fallback; never represented as a real checkpoint."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"status": "mock", "model_id": self.model_id, "input_keys": sorted(inputs)}

    def health(self) -> dict[str, Any]:
        return {"ready": True, "loaded": self.loaded, "mock": True}

    def metadata(self) -> dict[str, Any]:
        return {"id": self.model_id, "implementation": "Development Mock Result"}

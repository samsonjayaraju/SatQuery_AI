from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModelAdapter(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def health(self) -> dict[str, Any]: ...

    @abstractmethod
    def metadata(self) -> dict[str, Any]: ...

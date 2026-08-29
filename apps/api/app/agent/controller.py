from __future__ import annotations

from pathlib import Path
from collections.abc import Callable

from app.api.schemas.analysis import AnalysisResponse
from app.services.analysis_service import AnalysisService


class SatQueryAgent:
    """Thin orchestration boundary; task logic remains in replaceable services."""

    def __init__(self, analysis_service: AnalysisService):
        self.analysis_service = analysis_service

    def run(
        self,
        paths: list[Path],
        query: str,
        input_mode: str,
        progress: Callable[[str, str], None] | None = None,
    ) -> AnalysisResponse:
        try:
            return self.analysis_service.analyze(paths, query, input_mode, progress)
        finally:
            self.analysis_service.release_models()

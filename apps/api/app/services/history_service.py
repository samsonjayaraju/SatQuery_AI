from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.api.schemas.analysis import AnalysisResponse
from app.core.exceptions import SatQueryError


class HistoryService:
    def __init__(self, data_dir: Path):
        self.directory = data_dir.resolve() / "history"
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, result: AnalysisResponse) -> None:
        path = self.directory / f"{result.analysis_id}.json"
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    def get(self, analysis_id: str) -> AnalysisResponse:
        path = self.directory / f"{analysis_id}.json"
        if not path.exists():
            raise SatQueryError("ANALYSIS_NOT_FOUND", "No local analysis was found for that ID.", 404)
        return AnalysisResponse.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict[str, Any]]:
        records = []
        for path in sorted(self.directory.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                records.append(
                    {
                        "analysis_id": value["analysis_id"],
                        "created_at": value["created_at"],
                        "task": value["task"],
                        "query": value["query"],
                        "answer": value["answer"],
                        "confidence": value["confidence"]["overall"],
                    }
                )
            except (KeyError, json.JSONDecodeError):
                continue
        return records

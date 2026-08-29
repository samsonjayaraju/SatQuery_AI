from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_TASKS = (
    {"id": "single_image_vqa", "name": "Single Image VQA", "dataset": "VRSBench / RSVQA", "expected_metrics": ["accuracy", "exact_match"]},
    {"id": "change_detection", "name": "Change Detection", "dataset": "LEVIR-CD", "expected_metrics": ["iou", "f1", "precision", "recall"]},
    {"id": "change_vqa", "name": "Change VQA", "dataset": "CDVQA", "expected_metrics": ["accuracy", "exact_match"]},
    {"id": "visual_grounding", "name": "Visual Grounding", "dataset": "VRSBench", "expected_metrics": ["iou", "precision", "recall"]},
    {"id": "domain_adapter", "name": "Domain Adapter", "dataset": "BigEarthNet or configured open dataset", "expected_metrics": ["accuracy", "macro_f1"]},
)


class BenchmarkService:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def _results(self) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for path in sorted(self.directory.glob("**/*.json"), key=lambda item: item.stat().st_mtime):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                task_id = str(payload["task_id"])
                metrics = payload["metrics"]
                if not isinstance(metrics, dict) or not metrics:
                    continue
                payload["result_file"] = str(path.relative_to(self.directory))
                results[task_id] = payload
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return results

    def summary(self) -> dict[str, Any]:
        measured = self._results()
        tasks = []
        for definition in DEFAULT_TASKS:
            result = measured.get(definition["id"])
            tasks.append({**definition, "status": "measured" if result else "not_evaluated", "result": result})
        count = sum(task["status"] == "measured" for task in tasks)
        status = "measured" if count == len(tasks) else "partial" if count else "not_evaluated"
        return {
            "status": status,
            "message": f"{count} of {len(tasks)} benchmark tasks have measured local results.",
            "tasks": tasks,
        }

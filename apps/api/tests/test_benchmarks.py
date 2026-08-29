from __future__ import annotations

import json

from app.services.benchmark_service import BenchmarkService


def test_benchmarks_only_surface_measured_result_files(tmp_path):
    service = BenchmarkService(tmp_path)
    (tmp_path / "valid.json").write_text(
        json.dumps({"task_id": "change_detection", "model": "test", "metrics": {"iou": 0.71}}),
        encoding="utf-8",
    )
    (tmp_path / "invalid.json").write_text(json.dumps({"task_id": "single_image_vqa", "metrics": {}}), encoding="utf-8")

    summary = service.summary()
    assert summary["status"] == "partial"
    change = next(task for task in summary["tasks"] if task["id"] == "change_detection")
    vqa = next(task for task in summary["tasks"] if task["id"] == "single_image_vqa")
    assert change["result"]["metrics"]["iou"] == 0.71
    assert vqa["status"] == "not_evaluated"

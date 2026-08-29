from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from app.api.schemas.analysis import AnalysisJob, AnalysisResponse
from app.core.exceptions import SatQueryError
from app.services.file_service import cleanup_uploads


class AnalysisJobService:
    """Small in-process queue for local inference; no external broker is required."""

    def __init__(self, max_workers: int = 1):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="satquery-analysis")
        self._jobs: dict[str, AnalysisJob] = {}
        self._lock = threading.Lock()

    def _store(self, job: AnalysisJob) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def _update(self, job_id: str, status: str, message: str, **changes: object) -> None:
        progress_by_status = {
            "queued": 0.0,
            "validating": 0.12,
            "registering": 0.22,
            "loading_model": 0.3,
            "processing": 0.52,
            "postprocessing": 0.78,
            "integrating": 0.86,
            "completed": 1.0,
            "failed": 1.0,
        }
        with self._lock:
            current = self._jobs[job_id]
            payload = current.model_dump()
            payload.update(changes)
            payload.update(
                status=status,
                message=message,
                progress=progress_by_status.get(status, current.progress),
                updated_at=datetime.now(timezone.utc),
            )
            self._jobs[job_id] = AnalysisJob.model_validate(payload)

    def submit(
        self,
        paths: list[Path],
        query: str,
        input_mode: str,
        runner: Callable[[list[Path], str, str, Callable[[str, str], None]], AnalysisResponse],
    ) -> AnalysisJob:
        now = datetime.now(timezone.utc)
        job = AnalysisJob(
            job_id=str(uuid.uuid4()),
            status="queued",
            message="Queued for local analysis",
            created_at=now,
            updated_at=now,
        )
        self._store(job)
        self._executor.submit(self._execute, job.job_id, paths, query, input_mode, runner)
        return job

    def _execute(
        self,
        job_id: str,
        paths: list[Path],
        query: str,
        input_mode: str,
        runner: Callable[[list[Path], str, str, Callable[[str, str], None]], AnalysisResponse],
    ) -> None:
        try:
            self._update(job_id, "validating", "Validating local raster inputs")
            result = runner(paths, query, input_mode, lambda status, message: self._update(job_id, status, message))
            self._update(
                job_id,
                "completed",
                "Analysis completed",
                analysis_id=result.analysis_id,
                result=result,
            )
        except SatQueryError as exc:
            self._update(job_id, "failed", exc.message, error_code=exc.code)
        except Exception:
            self._update(
                job_id,
                "failed",
                "The local analysis failed unexpectedly. Check the API logs for details.",
                error_code="INTERNAL_ERROR",
            )
        finally:
            cleanup_uploads(paths)

    def get(self, job_id: str) -> AnalysisJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise SatQueryError("JOB_NOT_FOUND", "No local analysis job was found for that ID.", 404)
        return job.model_copy(deep=True)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

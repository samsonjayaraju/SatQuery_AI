from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import APIRouter, File, Form, Request, UploadFile, status
from PIL import Image

from app.api.schemas.analysis import AnalysisJob, AnalysisResponse, HealthResponse, InspectionResponse, ModelStatus, RegistrationInfo
from app.core.exceptions import SatQueryError
from app.models.manager import ModelManager
from app.remote_sensing.alignment import align_visual_pair
from app.remote_sensing.input_inspector import inspect_inputs
from app.remote_sensing.preprocessing import load_visual
from app.remote_sensing.visualization import save_image
from app.services.file_service import cleanup_uploads, save_uploads

router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    runtime = ModelManager.runtime_info()
    settings = request.app.state.settings
    models = request.app.state.model_registry.list()
    return HealthResponse(
        status="healthy",
        device=request.app.state.device,
        python_version=runtime["python"],
        pytorch_version=runtime["pytorch"],
        available_models=sum(model.status == "ready" for model in models),
        paths={"data": str(settings.data_dir.resolve()), "models": str(settings.model_dir.resolve())},
        mock_mode=settings.mock_mode,
    )


@router.get("/models", response_model=list[ModelStatus])
def models(request: Request) -> list[ModelStatus]:
    return request.app.state.model_registry.list()


@router.post("/files/inspect", response_model=InspectionResponse)
async def inspect_files(
    request: Request,
    input_mode: Annotated[str, Form()],
    files: Annotated[list[UploadFile], File()],
) -> InspectionResponse:
    request_id, paths = await save_uploads(files, request.app.state.settings)
    output = request.app.state.settings.data_dir.resolve() / "outputs" / f"inspection-{request_id}"
    try:
        response = inspect_inputs(paths, input_mode)
        alignment = None
        if len(paths) == 2 and response.valid:
            alignment = align_visual_pair(
                paths[0],
                paths[1],
                min_confidence=request.app.state.settings.registration_min_confidence,
            )
            source_arrays = [alignment.first, alignment.second]
            status_value = (
                "accepted"
                if alignment.confidence >= request.app.state.settings.registration_min_confidence
                else "low_quality"
            )
            warnings = list(alignment.warnings)
            if status_value == "low_quality":
                warnings.append("Input alignment quality is insufficient for highly confident quantitative change estimation.")
            response.registration = RegistrationInfo(
                method=alignment.method,
                confidence=alignment.confidence,
                status=status_value,
                transform=alignment.transform,
                warnings=warnings,
            )
            for warning in warnings:
                if warning not in response.warnings:
                    response.warnings.append(warning)
        else:
            source_arrays = [load_visual(path) for path in paths]

        for index, image in enumerate(source_arrays):
            max_side = max(image.shape[:2])
            if max_side > 760:
                scale = 760 / max_side
                image = np.asarray(
                    Image.fromarray(image).resize(
                        (round(image.shape[1] * scale), round(image.shape[0] * scale)),
                        Image.Resampling.BILINEAR,
                    )
                )
            preview = output / f"input-{index + 1}.png"
            save_image(image, preview)
            response.images[index].thumbnail_url = f"/assets/outputs/inspection-{request_id}/{preview.name}"
            response.images[index].display_width = source_arrays[index].shape[1]
            response.images[index].display_height = source_arrays[index].shape[0]
        return response
    finally:
        cleanup_uploads(paths)


async def _run_analysis(request: Request, query: str, input_mode: str, files: list[UploadFile]) -> AnalysisResponse:
    _, paths = await save_uploads(files, request.app.state.settings)
    try:
        return request.app.state.agent.run(paths, query, input_mode)
    finally:
        cleanup_uploads(paths)


@router.post("/analysis-jobs", response_model=AnalysisJob, status_code=status.HTTP_202_ACCEPTED)
async def create_analysis_job(
    request: Request,
    query: Annotated[str, Form(min_length=1, max_length=2000)],
    input_mode: Annotated[str, Form()],
    files: Annotated[list[UploadFile], File()],
) -> AnalysisJob:
    _, paths = await save_uploads(files, request.app.state.settings)
    try:
        inspection = inspect_inputs(paths, input_mode)
        if not inspection.valid:
            raise SatQueryError(
                "UNSUPPORTED_COMPOSITE_IMAGE",
                inspection.visual_quality.recommendation or "The selected input is not suitable for raster analysis.",
                422,
            )
        return request.app.state.jobs.submit(paths, query, input_mode, request.app.state.agent.run)
    except Exception:
        cleanup_uploads(paths)
        raise


@router.get("/analysis-jobs/{job_id}", response_model=AnalysisJob)
def get_analysis_job(request: Request, job_id: str) -> AnalysisJob:
    return request.app.state.jobs.get(job_id)


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    request: Request,
    query: Annotated[str, Form(min_length=1, max_length=2000)],
    input_mode: Annotated[str, Form()],
    files: Annotated[list[UploadFile], File()],
) -> AnalysisResponse:
    return await _run_analysis(request, query, input_mode, files)


@router.post("/analyze/single", response_model=AnalysisResponse)
async def analyze_single(
    request: Request,
    query: Annotated[str, Form(min_length=1, max_length=2000)],
    files: Annotated[list[UploadFile], File()],
) -> AnalysisResponse:
    return await _run_analysis(request, query, "single", files)


@router.post("/analyze/change", response_model=AnalysisResponse)
async def analyze_change(
    request: Request,
    query: Annotated[str, Form(min_length=1, max_length=2000)],
    files: Annotated[list[UploadFile], File()],
) -> AnalysisResponse:
    return await _run_analysis(request, query, "bi_temporal", files)


@router.post("/analyze/cross-modal", response_model=AnalysisResponse)
async def analyze_cross_modal(
    request: Request,
    query: Annotated[str, Form(min_length=1, max_length=2000)],
    files: Annotated[list[UploadFile], File()],
) -> AnalysisResponse:
    return await _run_analysis(request, query, "cross_modal", files)


@router.get("/history")
def list_history(request: Request):
    return request.app.state.history.list()


@router.get("/history/{analysis_id}", response_model=AnalysisResponse)
def get_history(request: Request, analysis_id: str) -> AnalysisResponse:
    return request.app.state.history.get(analysis_id)


@router.post("/reports/{analysis_id}")
def create_report(request: Request, analysis_id: str):
    result = request.app.state.history.get(analysis_id)
    path: Path = request.app.state.reports.build_html(result)
    return {"analysis_id": analysis_id, "format": "html", "report_url": f"/assets/reports/{path.name}"}


@router.get("/benchmarks")
def benchmarks(request: Request):
    return request.app.state.benchmarks.summary()

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, Request, UploadFile

from app.api.schemas.analysis import AnalysisResponse, HealthResponse, InspectionResponse, ModelStatus
from app.core.exceptions import SatQueryError
from app.models.manager import ModelManager
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
    urls = []
    for index, path in enumerate(paths):
        try:
            image = load_visual(path)
            max_side = max(image.shape[:2])
            if max_side > 760:
                scale = 760 / max_side
                image = load_visual(path, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
            preview = output / f"input-{index + 1}.png"
            save_image(image, preview)
            urls.append(f"/assets/outputs/inspection-{request_id}/{preview.name}")
        except Exception:
            urls.append(None)
    try:
        response = inspect_inputs(paths, input_mode, urls)
    except Exception:
        cleanup_uploads(paths)
        raise
    cleanup_uploads(paths)
    return response


async def _run_analysis(request: Request, query: str, input_mode: str, files: list[UploadFile]) -> AnalysisResponse:
    _, paths = await save_uploads(files, request.app.state.settings)
    try:
        return request.app.state.agent.run(paths, query, input_mode)
    finally:
        cleanup_uploads(paths)


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
def benchmarks():
    return {
        "status": "not_evaluated",
        "message": "Not evaluated yet.",
        "tasks": [
            {"name": "Single Image VQA", "dataset": "VRSBench / RSVQA", "metrics": None},
            {"name": "Change Detection", "dataset": "LEVIR-CD", "metrics": None},
            {"name": "Change VQA", "dataset": "CDVQA", "metrics": None},
            {"name": "Grounding", "dataset": "VRSBench", "metrics": None},
        ],
    }

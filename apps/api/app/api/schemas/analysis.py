from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


InputMode = Literal["single", "bi_temporal", "cross_modal"]


class RasterMetadata(BaseModel):
    filename: str
    file_size_bytes: int
    width: int
    height: int
    band_count: int
    data_type: str
    crs: str | None = None
    transform: list[float] | None = None
    bounds: list[float] | None = None
    pixel_resolution: list[float] | None = None
    nodata: float | None = None
    georeferenced: bool = False
    modality: str = "optical"
    format: str
    thumbnail_url: str | None = None
    display_width: int | None = None
    display_height: int | None = None


class Compatibility(BaseModel):
    crs_match: bool | None = None
    overlap: float | None = None
    co_registered: bool | None = None
    dimensions_match: bool | None = None
    resolution_compatible: bool | None = None
    warnings: list[str] = Field(default_factory=list)


class VisualQuality(BaseModel):
    status: Literal["accepted", "review", "unsupported"] = "accepted"
    score: float = 1.0
    flags: list[str] = Field(default_factory=list)
    recommendation: str | None = None


class RegistrationInfo(BaseModel):
    method: str = "not_required"
    confidence: float = 1.0
    status: Literal["not_required", "accepted", "low_quality"] = "not_required"
    transform: list[float] | None = None
    warnings: list[str] = Field(default_factory=list)


class InspectionResponse(BaseModel):
    valid: bool
    input_mode: InputMode
    images: list[RasterMetadata]
    compatibility: Compatibility
    visual_quality: VisualQuality = Field(default_factory=VisualQuality)
    registration: RegistrationInfo = Field(default_factory=RegistrationInfo)
    warnings: list[str] = Field(default_factory=list)


class IntentResult(BaseModel):
    intent: str
    confidence: float
    entities: dict[str, str] = Field(default_factory=dict)
    required_capabilities: list[str] = Field(default_factory=list)


class ConfidenceResult(BaseModel):
    overall: float
    type: Literal["model", "heuristic", "mixed"] = "heuristic"
    components: dict[str, float]
    note: str


class LegendItem(BaseModel):
    label: str
    color: str
    description: str | None = None


class EvidenceItem(BaseModel):
    id: str
    type: Literal[
        "mask",
        "overlay",
        "bounding_box",
        "polygon",
        "text",
        "heatmap",
        "probability_map",
        "transition",
        "statistic",
    ]
    label: str
    description: str
    confidence: float | None = None
    asset_url: str | None = None
    coordinates: list[list[float]] | None = None
    color: str = "#4fd1c5"
    legend: list[LegendItem] = Field(default_factory=list)


class TransitionItem(BaseModel):
    from_class: str
    to_class: str
    percent: float
    pixel_count: int
    area_square_metres: float | None = None
    area_hectares: float | None = None
    area_square_kilometres: float | None = None


class TraceStep(BaseModel):
    name: str
    kind: Literal["model", "tool", "system"]
    status: Literal["completed", "skipped", "failed"]
    runtime_ms: int
    detail: str


class ExecutionTrace(BaseModel):
    task: str
    input_mode: InputMode
    models: list[str]
    tools: list[str]
    parameters: dict[str, Any]
    steps: list[TraceStep]
    runtime_ms: int
    status: Literal["success", "failed"]
    mock_mode: bool
    warnings: list[str] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    analysis_id: str
    created_at: datetime
    task: str
    query: str
    answer: str
    development_label: str | None = None
    confidence: ConfidenceResult
    evidence: list[EvidenceItem]
    statistics: dict[str, float | int | str]
    transitions: list[TransitionItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    inspection: InspectionResponse
    execution_trace: ExecutionTrace
    runtime_ms: int


class AnalysisJob(BaseModel):
    job_id: str
    status: Literal[
        "queued",
        "validating",
        "registering",
        "loading_model",
        "processing",
        "postprocessing",
        "integrating",
        "completed",
        "failed",
    ]
    message: str
    progress: float = 0.0
    created_at: datetime
    updated_at: datetime
    analysis_id: str | None = None
    result: AnalysisResponse | None = None
    error_code: str | None = None


class ModelStatus(BaseModel):
    id: str
    name: str
    version: str
    status: Literal["ready", "checkpoint_missing", "disabled"]
    loaded: bool
    checkpoint_available: bool
    device: str
    supported_tasks: list[str]
    modalities: list[str]
    implementation: str
    checkpoint_path: str | None = None
    mode: Literal["real", "mock", "baseline", "disabled"]


class HealthResponse(BaseModel):
    status: str
    device: str
    python_version: str
    pytorch_version: str
    available_models: int
    paths: dict[str, str]
    mock_mode: bool

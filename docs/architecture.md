# Architecture

## Runtime surfaces

The Next.js client owns upload selection, raster metadata presentation, Leaflet visualization, assistant interaction and local report opening. The FastAPI server owns all filesystem access, validation, routing, analysis, evidence assets, confidence, history and model status.

```mermaid
sequenceDiagram
    actor User
    participant Web as Next.js Workspace
    participant API as FastAPI
    participant Inspector as InputInspector
    participant Agent as SatQueryAgent
    participant Specialists as Specialist tools/models
    participant Evidence as Evidence + Confidence
    User->>Web: Select mode, files and query
    Web->>API: POST /files/inspect
    API->>Inspector: Validate raster(s) and compatibility
    Inspector-->>Web: Metadata, warnings, thumbnails
    Web->>API: POST /analysis-jobs
    API-->>Web: 202 + job ID
    API->>Agent: Query + inspected mode in local worker
    Agent->>Agent: Classify intent and required capabilities
    Agent->>Specialists: Execute selected workflow
    Specialists-->>Evidence: Masks, probabilities, statistics
    loop Local polling
        Web->>API: GET /analysis-jobs/{id}
        API-->>Web: queued / validating / processing / integrating
    end
    Evidence-->>Web: completed result, assets, confidence, trace
```

## Agent contract

`SatQueryAgent` does not contain model-specific inference. It delegates to `AnalysisService`, which uses `QueryInterpreter`, `ModelRegistry`, `ToolRegistry`, modality-specific preprocessing, and the confidence/change reasoners. The active learned path uses RemoteCLIP RN50 plus the SatQuery EuroSAT adapter for single-image and optical evidence, and official ChangeFormer V6 for change probability. GeoChat and a future trained SatFusion head can replace those services without changing the API or frontend.

Only observable execution facts are returned: task, input mode, selected models/baselines, tools, parameters, step runtimes, status and warnings. Internal chain-of-thought is neither recorded nor exposed.

## Pipelines

```mermaid
flowchart TD
    Q[Query + inspection] --> R{Input mode}
    R -->|Single| S[Optical/SAR preprocessing]
    S --> VQA[VQA / caption / grounding adapter]
    S --> LC[Spectral land-cover evidence]
    R -->|Bi-temporal| A[Pixel alignment]
    A --> CD[ChangeFormer V6 adapter]
    CD --> CR[Before/after land-cover + ChangeReasoner]
    R -->|Optical + SAR| O[RemoteCLIP / adapter optical evidence]
    R -->|Optical + SAR| SAR[SAR backscatter evidence]
    O --> F[SatFusion weighted feature fusion]
    SAR --> F
    VQA --> E[EvidenceEngine]
    LC --> E
    CR --> E
    F --> E
    E --> C[ConfidenceEngine]
    C --> OUT[Answer + overlays + stats + trace + report]
```

Large rasters are not reduced to a single 1024-pixel analysis image. `TileWindow` creates overlapping full-resolution windows, each specialist produces tile-space probabilities, and weighted stitching restores them to source coordinates. Preview images are downsampled independently for the browser. Paired georeferenced rasters use Rasterio reprojection onto the first raster's CRS, affine grid and resolution; unreferenced inputs use an explicitly reported pixel-space resize fallback.

## Local job lifecycle

The in-process `AnalysisJobService` serializes memory-heavy runs through a single local worker. It exposes `queued`, `validating`, `loading_model`, `processing`, `integrating`, `completed`, and `failed` states without Redis or Celery. Upload copies remain request-scoped until their job finishes and are then removed in all success/failure paths. Completed results remain durable through history JSON and evidence assets.

## Storage

`data/uploads`, `outputs`, `reports`, `history` and `temp` are local and ignored. Analyses use UUIDs. History stores structured result metadata and asset references rather than duplicating original rasters. Uploaded files are never executed.

## Model lifecycle

Device priority is CUDA → Apple MPS → CPU, with an environment override. Registry entries report checkpoint and source readiness. RemoteCLIP and ChangeFormer expose lazy load/unload, prediction, health and metadata boundaries. With `MODEL_UNLOAD_AFTER_REQUEST=true`, the agent releases learned models even when analysis fails.

## Domain adaptation and evaluation

The training pipeline freezes RemoteCLIP RN50, caches features for a manifest-backed dataset, and trains a residual bottleneck plus task head. The completed EuroSAT run used 5,000 balanced samples and writes its best-epoch metrics to `evaluation-results/domain-adapter.json`. ChangeFormer evaluation uses the official upstream LEVIR demo list and writes `evaluation-results/change-detection.json`. The benchmark API reads only these files and never manufactures values for missing tasks.

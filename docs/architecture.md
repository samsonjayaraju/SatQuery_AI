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
    Web->>API: POST /analyze
    API->>Agent: Query + inspected mode
    Agent->>Agent: Classify intent and required capabilities
    Agent->>Specialists: Execute selected workflow
    Specialists-->>Evidence: Masks, probabilities, statistics
    Evidence-->>Web: Answer, assets, confidence, trace
```

## Agent contract

`SatQueryAgent` does not contain model-specific inference. It delegates to `AnalysisService`, which uses `QueryInterpreter`, `ModelRegistry`, `ToolRegistry`, modality-specific preprocessing, and the confidence/change reasoners. This preserves the API when a deterministic baseline is replaced by GeoChat, RemoteCLIP, ChangeFormer or a trained SatFusion head.

Only observable execution facts are returned: task, input mode, selected models/baselines, tools, parameters, step runtimes, status and warnings. Internal chain-of-thought is neither recorded nor exposed.

## Pipelines

```mermaid
flowchart TD
    Q[Query + inspection] --> R{Input mode}
    R -->|Single| S[Optical/SAR preprocessing]
    S --> VQA[VQA / caption / grounding adapter]
    S --> LC[Spectral land-cover evidence]
    R -->|Bi-temporal| A[Pixel alignment]
    A --> CD[Change detector adapter]
    CD --> CR[Before/after land-cover + ChangeReasoner]
    R -->|Optical + SAR| O[Optical evidence]
    R -->|Optical + SAR| SAR[SAR backscatter evidence]
    O --> F[SatFusion]
    SAR --> F
    VQA --> E[EvidenceEngine]
    LC --> E
    CR --> E
    F --> E
    E --> C[ConfidenceEngine]
    C --> OUT[Answer + overlays + stats + trace + report]
```

## Storage

`data/uploads`, `outputs`, `reports`, `history` and `temp` are local and ignored. Analyses use UUIDs. History stores structured result metadata and asset references rather than duplicating original rasters. Uploaded files are never executed.

## Model lifecycle

Device priority is CUDA → Apple MPS → CPU, with an environment override. Registry entries report checkpoint readiness. The adapter boundary defines `load`, `unload`, `predict`, `health` and `metadata`. Large learned adapters can be loaded lazily and unloaded after each request.

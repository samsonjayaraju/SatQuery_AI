export type InputMode = "single" | "bi_temporal" | "cross_modal";
export type AnalysisStatus = "empty" | "validating" | "ready" | "processing" | "completed" | "failed";

export interface RasterMetadata {
  filename: string;
  file_size_bytes: number;
  width: number;
  height: number;
  band_count: number;
  data_type: string;
  crs: string | null;
  transform: number[] | null;
  bounds: number[] | null;
  pixel_resolution: number[] | null;
  nodata: number | null;
  georeferenced: boolean;
  modality: string;
  format: string;
  thumbnail_url: string | null;
}

export interface InspectionResponse {
  valid: boolean;
  input_mode: InputMode;
  images: RasterMetadata[];
  compatibility: {
    crs_match: boolean | null;
    overlap: number | null;
    co_registered: boolean | null;
    dimensions_match: boolean | null;
    resolution_compatible: boolean | null;
    warnings: string[];
  };
  warnings: string[];
}

export interface EvidenceItem {
  id: string;
  type: "mask" | "overlay" | "bounding_box" | "polygon" | "text" | "heatmap";
  label: string;
  description: string;
  confidence: number | null;
  asset_url: string | null;
  coordinates: number[][] | null;
  color: string;
}

export interface TraceStep {
  name: string;
  kind: "model" | "tool" | "system";
  status: "completed" | "skipped" | "failed";
  runtime_ms: number;
  detail: string;
}

export interface AnalysisResponse {
  analysis_id: string;
  created_at: string;
  task: string;
  query: string;
  answer: string;
  development_label: string | null;
  confidence: {
    overall: number;
    type: "model" | "heuristic" | "mixed";
    components: Record<string, number>;
    note: string;
  };
  evidence: EvidenceItem[];
  statistics: Record<string, string | number>;
  inspection: InspectionResponse;
  execution_trace: {
    task: string;
    input_mode: InputMode;
    models: string[];
    tools: string[];
    parameters: Record<string, string | number | boolean>;
    steps: TraceStep[];
    runtime_ms: number;
    status: "success" | "failed";
    mock_mode: boolean;
  };
  runtime_ms: number;
}

export interface HealthResponse {
  status: string;
  device: string;
  python_version: string;
  pytorch_version: string;
  available_models: number;
  paths: Record<string, string>;
  mock_mode: boolean;
}

export interface ModelStatus {
  id: string;
  name: string;
  version: string;
  status: "ready" | "checkpoint_missing" | "disabled";
  loaded: boolean;
  checkpoint_available: boolean;
  device: string;
  supported_tasks: string[];
  modalities: string[];
  implementation: string;
}

export interface HistoryItem {
  analysis_id: string;
  created_at: string;
  task: string;
  query: string;
  answer: string;
  confidence: number;
}

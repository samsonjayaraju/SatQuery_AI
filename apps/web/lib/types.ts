export type InputMode = "single" | "bi_temporal" | "cross_modal";
export type JobStatus = "queued" | "validating" | "registering" | "loading_model" | "processing" | "postprocessing" | "integrating" | "completed" | "failed";
export type AnalysisStatus = "empty" | "uploading" | "ready" | JobStatus;

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
  display_width: number | null;
  display_height: number | null;
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
  visual_quality: {
    status: "accepted" | "review" | "unsupported";
    score: number;
    flags: string[];
    recommendation: string | null;
  };
  registration: {
    method: string;
    confidence: number;
    status: "not_required" | "accepted" | "low_quality";
    transform: number[] | null;
    warnings: string[];
  };
  warnings: string[];
}

export interface EvidenceItem {
  id: string;
  type: "mask" | "overlay" | "bounding_box" | "polygon" | "text" | "heatmap" | "probability_map" | "transition" | "statistic";
  label: string;
  description: string;
  confidence: number | null;
  asset_url: string | null;
  coordinates: number[][] | null;
  color: string;
  legend: { label: string; color: string; description: string | null }[];
}

export interface TransitionItem {
  from_class: string;
  to_class: string;
  percent: number;
  pixel_count: number;
  area_square_metres: number | null;
  area_hectares: number | null;
  area_square_kilometres: number | null;
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
  transitions: TransitionItem[];
  warnings: string[];
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
    warnings: string[];
  };
  runtime_ms: number;
}

export interface AnalysisJob {
  job_id: string;
  status: JobStatus;
  message: string;
  progress: number;
  created_at: string;
  updated_at: string;
  analysis_id: string | null;
  result: AnalysisResponse | null;
  error_code: string | null;
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
  checkpoint_path: string | null;
  mode: "real" | "mock" | "baseline" | "disabled";
}

export interface HistoryItem {
  analysis_id: string;
  created_at: string;
  task: string;
  query: string;
  answer: string;
  confidence: number;
}

export interface BenchmarkTask {
  id: string;
  name: string;
  dataset: string;
  expected_metrics: string[];
  status: "measured" | "not_evaluated";
  result: null | {
    dataset?: string;
    model?: string;
    split?: string;
    sample_count?: number;
    created_at?: string;
    result_file: string;
    metrics: Record<string, number>;
  };
}

export interface BenchmarkResponse {
  status: "measured" | "partial" | "not_evaluated";
  message: string;
  tasks: BenchmarkTask[];
}

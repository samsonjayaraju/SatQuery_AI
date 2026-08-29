import type {
  AnalysisResponse,
  AnalysisJob,
  BenchmarkResponse,
  HealthResponse,
  HistoryItem,
  InputMode,
  InspectionResponse,
  ModelStatus,
} from "@/lib/types";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.error?.message ?? `Local API request failed (${response.status}).`);
  }
  return response.json() as Promise<T>;
}

function fileForm(files: File[], inputMode: InputMode, query?: string): FormData {
  const body = new FormData();
  body.append("input_mode", inputMode);
  if (query) body.append("query", query);
  files.forEach((file) => body.append("files", file));
  return body;
}

export function assetUrl(path: string | null | undefined): string | null {
  if (!path) return null;
  return path.startsWith("http") ? path : `${API_URL}${path}`;
}

export async function inspectFiles(files: File[], inputMode: InputMode): Promise<InspectionResponse> {
  return parseResponse(
    await fetch(`${API_URL}/api/v1/files/inspect`, {
      method: "POST",
      body: fileForm(files, inputMode),
    }),
  );
}

export async function analyzeFiles(files: File[], inputMode: InputMode, query: string): Promise<AnalysisResponse> {
  return parseResponse(
    await fetch(`${API_URL}/api/v1/analyze`, {
      method: "POST",
      body: fileForm(files, inputMode, query),
    }),
  );
}

export async function createAnalysisJob(files: File[], inputMode: InputMode, query: string): Promise<AnalysisJob> {
  return parseResponse(
    await fetch(`${API_URL}/api/v1/analysis-jobs`, {
      method: "POST",
      body: fileForm(files, inputMode, query),
    }),
  );
}

export async function fetchAnalysisJob(jobId: string): Promise<AnalysisJob> {
  return parseResponse(await fetch(`${API_URL}/api/v1/analysis-jobs/${jobId}`, { cache: "no-store" }));
}

export async function waitForAnalysisJob(
  jobId: string,
  onProgress?: (job: AnalysisJob) => void,
  pollIntervalMs = 450,
): Promise<AnalysisResponse> {
  for (let attempt = 0; attempt < 4000; attempt += 1) {
    const job = await fetchAnalysisJob(jobId);
    onProgress?.(job);
    if (job.status === "completed" && job.result) return job.result;
    if (job.status === "failed") throw new Error(job.message);
    await new Promise((resolve) => window.setTimeout(resolve, pollIntervalMs));
  }
  throw new Error("The local analysis timed out before completion.");
}

export async function fetchHealth(): Promise<HealthResponse> {
  return parseResponse(await fetch(`${API_URL}/api/v1/health`, { cache: "no-store" }));
}

export async function fetchModels(): Promise<ModelStatus[]> {
  return parseResponse(await fetch(`${API_URL}/api/v1/models`, { cache: "no-store" }));
}

export async function fetchHistory(): Promise<HistoryItem[]> {
  return parseResponse(await fetch(`${API_URL}/api/v1/history`, { cache: "no-store" }));
}

export async function fetchBenchmarks(): Promise<BenchmarkResponse> {
  return parseResponse(await fetch(`${API_URL}/api/v1/benchmarks`, { cache: "no-store" }));
}

export async function fetchAnalysis(id: string): Promise<AnalysisResponse> {
  return parseResponse(await fetch(`${API_URL}/api/v1/history/${id}`, { cache: "no-store" }));
}

export async function createReport(id: string): Promise<string> {
  const result = await parseResponse<{ report_url: string }>(
    await fetch(`${API_URL}/api/v1/reports/${id}`, { method: "POST" }),
  );
  return assetUrl(result.report_url)!;
}

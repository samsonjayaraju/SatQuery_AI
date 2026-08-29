"use client";

import { useMemo, useState } from "react";
import { AppHeader } from "@/components/shell/app-header";
import { createAnalysisJob, createReport, inspectFiles, waitForAnalysisJob } from "@/lib/api";
import type { AnalysisJob, AnalysisResponse, AnalysisStatus, InputMode, InspectionResponse } from "@/lib/types";
import { AssistantPanel } from "./assistant-panel";
import { ExecutionTrace } from "./execution-trace";
import { FileInputPanel } from "./file-input-panel";
import { ViewerPanel } from "./viewer-panel";

const defaultQueries: Record<InputMode, string> = {
  single: "Describe the land-cover and major objects visible.",
  bi_temporal: "What changed between these two dates?",
  cross_modal: "Use both sensors to identify water-covered areas.",
};

export function WorkspaceApp() {
  const [mode, setMode] = useState<InputMode>("single");
  const [files, setFiles] = useState<File[]>([]);
  const [inspection, setInspection] = useState<InspectionResponse | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [status, setStatus] = useState<AnalysisStatus>("empty");
  const [query, setQuery] = useState(defaultQueries.single);
  const [error, setError] = useState<string | null>(null);
  const [traceOpen, setTraceOpen] = useState(false);
  const [activeEvidence, setActiveEvidence] = useState(0);
  const [activeImage, setActiveImage] = useState(0);
  const [split, setSplit] = useState(false);
  const [compare, setCompare] = useState(false);
  const [overlayVisible, setOverlayVisible] = useState(true);
  const [overlayOpacity, setOverlayOpacity] = useState(0.78);
  const [resetKey, setResetKey] = useState(0);
  const [reportBusy, setReportBusy] = useState(false);
  const [job, setJob] = useState<AnalysisJob | null>(null);

  const expected = mode === "single" ? 1 : 2;
  const complete = files.length === expected;

  async function inspect(selected: File[], selectedMode: InputMode) {
    const count = selectedMode === "single" ? 1 : 2;
    if (selected.length !== count) {
      setInspection(null);
      setStatus(selected.length ? "empty" : "empty");
      return;
    }
    setStatus("validating");
    setError(null);
    try {
      const inspected = await inspectFiles(selected, selectedMode);
      setInspection(inspected);
      setStatus("ready");
    } catch (reason) {
      setInspection(null);
      setStatus("failed");
      setError(reason instanceof Error ? reason.message : "Input validation failed.");
    }
  }

  async function handleFiles(selected: File[]) {
    setFiles(selected);
    setResult(null);
    setActiveImage(0);
    setSplit(false);
    setCompare(false);
    setOverlayVisible(true);
    if (selected.length) await inspect(selected, mode);
    else {
      setInspection(null);
      setStatus("empty");
      setError(null);
    }
  }

  function changeMode(next: InputMode) {
    setMode(next);
    setFiles([]);
    setInspection(null);
    setResult(null);
    setStatus("empty");
    setError(null);
    setQuery(defaultQueries[next]);
    setActiveImage(0);
    setSplit(false);
    setCompare(false);
    setOverlayVisible(true);
    setTraceOpen(false);
    setJob(null);
  }

  async function analyze() {
    if (!complete || !query.trim() || ["queued", "loading_model", "processing", "integrating"].includes(status)) return;
    setStatus("queued");
    setError(null);
    setResult(null);
    setTraceOpen(false);
    try {
      const created = await createAnalysisJob(files, mode, query.trim());
      setJob(created);
      const next = await waitForAnalysisJob(created.job_id, (nextJob) => {
        setJob(nextJob);
        setStatus(nextJob.status);
      });
      setResult(next);
      setInspection(next.inspection);
      setStatus("completed");
      setActiveEvidence(0);
      setActiveImage(mode === "bi_temporal" ? 1 : 0);
      setOverlayOpacity(.78);
      setOverlayVisible(true);
    } catch (reason) {
      setStatus("failed");
      setError(reason instanceof Error ? reason.message : "Analysis failed.");
    }
  }

  async function report() {
    if (!result || reportBusy) return;
    setReportBusy(true);
    try {
      const url = await createReport(result.analysis_id);
      window.open(url, "_blank", "noopener,noreferrer");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Report generation failed.");
    } finally {
      setReportBusy(false);
    }
  }

  const appClass = useMemo(() => `app-frame ${traceOpen && result ? "trace-open" : ""}`, [traceOpen, result]);

  return (
    <main className={appClass}>
      <AppHeader active="workspace" />
      <section className="workspace-grid">
        <FileInputPanel mode={mode} files={files} inspection={inspection} status={status} error={error} onModeChange={changeMode} onFiles={(next) => { void handleFiles(next); }} />
        <ViewerPanel inspection={inspection} result={result} activeEvidence={activeEvidence} overlayOpacity={overlayOpacity} activeImage={activeImage} split={split} compare={compare} overlayVisible={overlayVisible} resetKey={resetKey} onOpacity={setOverlayOpacity} onImage={setActiveImage} onSplit={() => { setSplit((value) => !value); setCompare(false); }} onCompare={() => { setCompare((value) => !value); setSplit(false); }} onOverlay={() => setOverlayVisible((value) => !value)} onFit={() => setResetKey((value) => value + 1)} />
        <AssistantPanel mode={mode} status={status} job={job} query={query} result={result} activeEvidence={activeEvidence} canAnalyze={complete && Boolean(inspection?.valid)} reportBusy={reportBusy} onQuery={setQuery} onAnalyze={analyze} onEvidence={setActiveEvidence} onReport={report} />
      </section>
      <ExecutionTrace result={result} open={traceOpen} onToggle={() => setTraceOpen((value) => !value)} />
    </main>
  );
}

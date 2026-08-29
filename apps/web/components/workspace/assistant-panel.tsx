"use client";

import { AlertTriangle, Bot, Check, ChevronsUp, Download, LoaderCircle, MessageSquareText, Sparkles } from "lucide-react";
import { FormEvent } from "react";
import type { AnalysisJob, AnalysisResponse, AnalysisStatus, InputMode } from "@/lib/types";
import { Button } from "@/components/ui/button";

const prompts: Record<InputMode, string[]> = {
  single: [
    "Describe the land-cover and major objects visible.",
    "Highlight the largest water body.",
    "What percentage of this area appears to be vegetation?",
  ],
  bi_temporal: [
    "What changed between these two dates?",
    "Has the built-up area increased?",
    "Where did the main change occur?",
  ],
  cross_modal: [
    "Use both sensors to identify water-covered areas.",
    "Where do optical and SAR evidence agree?",
    "Identify built-up regions using both sensors.",
  ],
};

function readableStat(value: string | number, key: string) {
  if (typeof value === "string") return value.replaceAll("_", " ");
  if (key.includes("percent") || key.endsWith("_pp")) return `${value.toFixed(1)}${key.endsWith("_pp") ? " pp" : "%"}`;
  if (key.includes("probability")) return value.toFixed(3);
  return String(value);
}

const confidenceLabels: Record<string, string> = {
  evidence_strength: "Model evidence",
  model_score: "Change model",
  registration_quality: "Registration quality",
  spatial_quality: "Spatial evidence",
  cross_sensor_agreement: "Cross-sensor agreement",
  semantic_consistency: "Land-cover agreement",
  input_quality: "Input quality",
};

export function AssistantPanel({
  mode,
  status,
  job,
  query,
  result,
  activeEvidence,
  canAnalyze,
  reportBusy,
  onQuery,
  onAnalyze,
  onEvidence,
  onReport,
}: {
  mode: InputMode;
  status: AnalysisStatus;
  job: AnalysisJob | null;
  query: string;
  result: AnalysisResponse | null;
  activeEvidence: number;
  canAnalyze: boolean;
  reportBusy: boolean;
  onQuery: (query: string) => void;
  onAnalyze: () => void;
  onEvidence: (index: number) => void;
  onReport: () => void;
}) {
  const busy = ["queued", "validating", "registering", "loading_model", "processing", "postprocessing", "integrating"].includes(status);
  function submit(event: FormEvent) {
    event.preventDefault();
    onAnalyze();
  }

  return (
    <aside className="assistant-panel">
      <div className="panel-heading">
        <div><span className="eyebrow">02 · SATQUERY AGENT</span><h1>Ask the landscape</h1></div>
        <span className={`agent-status ${busy ? "busy" : ""}`}><span /> {busy ? status.replaceAll("_", " ").toUpperCase() : "READY"}</span>
      </div>

      {!result ? (
        <>
          <div className="assistant-intro">
            <span><Bot size={22} /></span>
            <h2>Sensor-aware analysis</h2>
            <p>I route each question to a specialist workflow, then return the answer with spatial evidence and an auditable trace.</p>
          </div>
          <div className="prompt-list">
            {prompts[mode].map((prompt) => (
              <button type="button" key={prompt} onClick={() => onQuery(prompt)}><span>{prompt}</span><ChevronsUp size={15} /></button>
            ))}
          </div>
          {busy && (
            <div className="processing-card"><LoaderCircle className="spin" size={20} /><div><b>{job?.message ?? "Preparing local analysis"}</b><span>{Math.round((job?.progress ?? 0) * 100)}% · {status.replaceAll("_", " ")}</span><i style={{ width: `${Math.round((job?.progress ?? 0) * 100)}%` }} /></div></div>
          )}
        </>
      ) : (
        <div className="result-scroll">
          {result.development_label && <div className="mock-label"><AlertTriangle size={13} /> {result.development_label}</div>}
          {result.warnings.length > 0 && <div className="result-warning"><AlertTriangle size={14} /><span><b>ANALYSIS ADVISORY</b>{result.warnings.join(" ")}</span></div>}
          <div className="user-query"><span>QUERY</span><p>{result.query}</p></div>
          <article className="answer-card">
            <div className="answer-heading"><span><Bot size={15} /> SATQUERY</span><span>{result.task.replaceAll("_", " ")}</span></div>
            <p>{result.answer}</p>
          </article>
          <section className="confidence-card">
            <div className="confidence-ring" style={{ "--score": `${result.confidence.overall * 360}deg` } as React.CSSProperties}><span>{Math.round(result.confidence.overall * 100)}<small>%</small></span></div>
            <div><span className="eyebrow">CONFIDENCE · {result.confidence.type.toUpperCase()}</span><strong>{result.confidence.overall >= .75 ? "Strong evidence" : result.confidence.overall >= .55 ? "Moderate evidence" : "Review advised"}</strong><small>{result.confidence.note}</small></div>
          </section>
          <div className="confidence-components">
            {Object.entries(result.confidence.components).map(([key, value]) => <div key={key}><span>{confidenceLabels[key] ?? key.replaceAll("_", " ")}</span><b>{Math.round(value * 100)}%</b></div>)}
          </div>
          <section className="model-provenance">
            <span className="eyebrow">MODELS / BASELINES</span>
            <div>{result.execution_trace.models.map((model) => <span key={model}>{model}</span>)}</div>
          </section>
          <section className="result-section">
            <div className="section-label"><span className="eyebrow">SPATIAL EVIDENCE</span><span>{result.evidence.length} LAYERS</span></div>
            <div className="evidence-list">
              {result.evidence.map((item, index) => (
                <button type="button" className={activeEvidence === index ? "active" : ""} aria-pressed={activeEvidence === index} key={item.id} onClick={() => onEvidence(index)}>
                  <i style={{ background: item.color }} /><span><b>{item.label}</b><small>{item.description}</small></span><Check size={13} />
                </button>
              ))}
            </div>
          </section>
          {result.transitions.length > 0 && <section className="result-section transition-section">
            <span className="eyebrow">LAND-COVER TRANSITIONS</span>
            <div>{result.transitions.slice(0, 6).map((item) => <p key={`${item.from_class}-${item.to_class}`}><span>{item.from_class === "unchanged" ? "Unchanged" : `${item.from_class.replaceAll("_", " ")} → ${item.to_class.replaceAll("_", " ")}`}</span><b>{item.percent.toFixed(1)}%</b></p>)}</div>
          </section>}
          <section className="result-section stats-section">
            <span className="eyebrow">MEASUREMENTS</span>
            <div className="stat-grid">
              {Object.entries(result.statistics).slice(0, 8).map(([key, value]) => (
                <div key={key}><span>{key.replaceAll("_", " ")}</span><b>{readableStat(value, key)}</b></div>
              ))}
            </div>
          </section>
          <Button className="report-button" variant="outline" size="sm" type="button" onClick={onReport} disabled={reportBusy}>
            {reportBusy ? <LoaderCircle className="spin" size={15} /> : <Download size={15} />} Generate local report
          </Button>
        </div>
      )}

      <form className="query-composer" onSubmit={submit}>
        <textarea aria-label="Analysis question" placeholder={canAnalyze ? "Ask a question about the imagery…" : "Load valid imagery to begin…"} value={query} onChange={(event) => onQuery(event.target.value)} disabled={!canAnalyze || busy} />
        <div><span><Sparkles size={13} /> Automatic routing</span><button type="submit" disabled={!canAnalyze || !query.trim() || busy}>{busy ? <LoaderCircle className="spin" size={16} /> : <MessageSquareText size={16} />} Analyze</button></div>
      </form>
    </aside>
  );
}

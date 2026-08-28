"use client";

import { AlertTriangle, Bot, Check, ChevronsUp, Download, LoaderCircle, MessageSquareText, Sparkles } from "lucide-react";
import { FormEvent } from "react";
import type { AnalysisResponse, AnalysisStatus, InputMode } from "@/lib/types";
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

export function AssistantPanel({
  mode,
  status,
  query,
  result,
  canAnalyze,
  reportBusy,
  onQuery,
  onAnalyze,
  onEvidence,
  onReport,
}: {
  mode: InputMode;
  status: AnalysisStatus;
  query: string;
  result: AnalysisResponse | null;
  canAnalyze: boolean;
  reportBusy: boolean;
  onQuery: (query: string) => void;
  onAnalyze: () => void;
  onEvidence: (index: number) => void;
  onReport: () => void;
}) {
  function submit(event: FormEvent) {
    event.preventDefault();
    onAnalyze();
  }

  return (
    <aside className="assistant-panel">
      <div className="panel-heading">
        <div><span className="eyebrow">02 · SATQUERY AGENT</span><h1>Ask the landscape</h1></div>
        <span className={`agent-status ${status === "processing" ? "busy" : ""}`}><span /> {status === "processing" ? "ANALYZING" : "READY"}</span>
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
          {status === "processing" && (
            <div className="processing-card"><LoaderCircle className="spin" size={20} /><div><b>Specialists are working</b><span>Validating · routing · extracting evidence</span></div></div>
          )}
        </>
      ) : (
        <div className="result-scroll">
          {result.development_label && <div className="mock-label"><AlertTriangle size={13} /> {result.development_label}</div>}
          <div className="user-query"><span>QUERY</span><p>{result.query}</p></div>
          <article className="answer-card">
            <div className="answer-heading"><span><Bot size={15} /> SATQUERY</span><span>{result.task.replaceAll("_", " ")}</span></div>
            <p>{result.answer}</p>
          </article>
          <section className="confidence-card">
            <div className="confidence-ring" style={{ "--score": `${result.confidence.overall * 360}deg` } as React.CSSProperties}><span>{Math.round(result.confidence.overall * 100)}<small>%</small></span></div>
            <div><span className="eyebrow">CONFIDENCE · {result.confidence.type.toUpperCase()}</span><strong>{result.confidence.overall >= .75 ? "Strong evidence" : result.confidence.overall >= .55 ? "Moderate evidence" : "Review advised"}</strong><small>{result.confidence.note}</small></div>
          </section>
          <section className="result-section">
            <div className="section-label"><span className="eyebrow">SPATIAL EVIDENCE</span><span>{result.evidence.length} LAYERS</span></div>
            <div className="evidence-list">
              {result.evidence.map((item, index) => (
                <button type="button" key={item.id} onClick={() => onEvidence(index)}>
                  <i style={{ background: item.color }} /><span><b>{item.label}</b><small>{item.description}</small></span><Check size={13} />
                </button>
              ))}
            </div>
          </section>
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
        <textarea aria-label="Analysis question" placeholder={canAnalyze ? "Ask a question about the imagery…" : "Load valid imagery to begin…"} value={query} onChange={(event) => onQuery(event.target.value)} disabled={!canAnalyze || status === "processing"} />
        <div><span><Sparkles size={13} /> Automatic routing</span><button type="submit" disabled={!canAnalyze || !query.trim() || status === "processing"}>{status === "processing" ? <LoaderCircle className="spin" size={16} /> : <MessageSquareText size={16} />} Analyze</button></div>
      </form>
    </aside>
  );
}

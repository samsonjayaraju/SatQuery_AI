"use client";

import { Activity, AlertTriangle, Check, ChevronDown, ChevronsUp, Cpu, Wrench } from "lucide-react";
import type { AnalysisResponse } from "@/lib/types";

export function ExecutionTrace({ result, open, onToggle }: { result: AnalysisResponse | null; open: boolean; onToggle: () => void }) {
  return (
    <>
      {open && result && (
        <section className="trace-drawer" aria-label="Execution trace details">
          <div className="trace-summary">
            <div><span className="eyebrow">DETECTED TASK</span><b>{result.task.replaceAll("_", " ")}</b></div>
            <div><span className="eyebrow">INPUT MODE</span><b>{result.execution_trace.input_mode.replaceAll("_", " ")}</b></div>
            <div><span className="eyebrow">RUNTIME</span><b>{result.runtime_ms} ms</b></div>
            <div><span className="eyebrow">STATUS</span><b className="success"><Check size={12} /> Success</b></div>
          </div>
          <div className="trace-content">
            <div className="trace-steps">
              {result.execution_trace.steps.map((step, index) => (
                <div className="trace-step" key={`${step.name}-${index}`}>
                  <span className="step-marker"><Check size={11} /></span>
                  <div><b>{step.name}</b><small>{step.detail}</small></div><time>{step.runtime_ms} ms</time>
                </div>
              ))}
            </div>
            <div className="trace-resources">
              <div><span><Cpu size={13} /> Models / baselines</span>{result.execution_trace.models.map((model) => <b key={model}>{model}</b>)}</div>
              <div><span><Wrench size={13} /> Tools</span><p>{result.execution_trace.tools.join(" · ")}</p></div>
              <div><span>Parameters</span><p>{Object.entries(result.execution_trace.parameters).map(([key, value]) => `${key.replaceAll("_", " ")}: ${value}`).join(" · ")}</p></div>
              {result.execution_trace.warnings.length > 0 && <div className="trace-warnings"><span><AlertTriangle size={13} /> Warnings</span><p>{result.execution_trace.warnings.join(" ")}</p></div>}
            </div>
          </div>
        </section>
      )}
      <footer className="trace-bar">
        <div><Activity size={15} /><span><b>Execution trace</b>{result ? `${result.execution_trace.models.length} specialists · ${result.execution_trace.tools.length} tools · ${result.runtime_ms} ms` : "No analysis has run in this session."}</span></div>
        <button type="button" onClick={onToggle} disabled={!result}>{open ? "Collapse" : "Expand"} {open ? <ChevronDown size={15} /> : <ChevronsUp size={15} />}</button>
      </footer>
    </>
  );
}

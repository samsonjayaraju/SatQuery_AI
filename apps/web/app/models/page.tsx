"use client";

import { useQuery } from "@tanstack/react-query";
import { Box, Check, CircleOff, LoaderCircle } from "lucide-react";
import { SectionPage } from "@/components/shell/section-page";
import { fetchModels } from "@/lib/api";

export default function ModelsPage() {
  const models = useQuery({ queryKey: ["models"], queryFn: fetchModels });
  return (
    <SectionPage active="models" eyebrow="SPECIALIST REGISTRY" title="Models & baselines" description="Checkpoint readiness, devices and supported tasks are reported by the local model registry—not hardcoded in the interface.">
      {models.isLoading ? <div className="page-empty"><LoaderCircle className="spin" /> Inspecting the registry…</div> : models.isError ? <div className="page-empty error">The API is offline. Start FastAPI to inspect local checkpoints.</div> : (
        <div className="model-grid">
          {models.data?.map((model) => (
            <article key={model.id}>
              <header><span className="model-icon"><Box size={17} /></span><span className={`model-state ${model.status}`} >{model.status === "ready" ? <Check size={11} /> : <CircleOff size={11} />}{model.status.replaceAll("_", " ")}</span></header>
              <h2>{model.name}</h2><p>{model.id} · {model.version}</p>
              <dl><div><dt>Implementation</dt><dd>{model.implementation.replaceAll("_", " ")}</dd></div><div><dt>Device</dt><dd>{model.device.toUpperCase()}</dd></div><div><dt>Loaded</dt><dd>{model.loaded ? "Yes" : "Lazy"}</dd></div></dl>
              <div className="chip-list">{model.supported_tasks.map((task) => <span key={task}>{task.replaceAll("_", " ")}</span>)}</div>
            </article>
          ))}
        </div>
      )}
    </SectionPage>
  );
}

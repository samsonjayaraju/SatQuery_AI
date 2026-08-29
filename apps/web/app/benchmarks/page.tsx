"use client";

import { useQuery } from "@tanstack/react-query";
import { Beaker, CheckCircle2, CircleDashed, LoaderCircle } from "lucide-react";
import { SectionPage } from "@/components/shell/section-page";
import { fetchBenchmarks } from "@/lib/api";

function formatMetric(value: number) {
  return value >= 0 && value <= 1 ? `${(value * 100).toFixed(1)}%` : value.toFixed(3);
}

export default function BenchmarksPage() {
  const benchmarks = useQuery({ queryKey: ["benchmarks"], queryFn: fetchBenchmarks });
  return (
    <SectionPage active="benchmarks" eyebrow="MEASURABLE EVALUATION" title="Benchmarks" description="This page only displays metrics produced by local evaluation result files. Placeholder scores are intentionally prohibited.">
      {benchmarks.isLoading ? <div className="page-empty"><LoaderCircle className="spin" /> Reading evaluation files…</div> : benchmarks.isError ? <div className="page-empty error">The API is offline. Start FastAPI to read measured results.</div> : <>
        <div className="benchmark-note"><Beaker size={18} /><div><b>{benchmarks.data?.message}</b><span>Only JSON files produced by local evaluation scripts are accepted; missing tasks remain explicitly unevaluated.</span></div></div>
        <div className="benchmark-grid">
          {benchmarks.data?.tasks.map((task) => <article key={task.id}><header>{task.result ? <CheckCircle2 size={17} /> : <CircleDashed size={17} />}<span>{task.result ? "MEASURED" : "AWAITING RUN"}</span></header><h2>{task.name}</h2><p>{task.dataset}</p><small>{task.expected_metrics.join(" · ")}</small>{task.result ? <div className="measured-metrics">{Object.entries(task.result.metrics).map(([name, value]) => <div key={name}><span>{name.replaceAll("_", " ")}</span><b>{formatMetric(value)}</b></div>)}</div> : <div className="metric-placeholder"><span>—</span><b>No measured result</b></div>}</article>)}
        </div>
      </>}
    </SectionPage>
  );
}

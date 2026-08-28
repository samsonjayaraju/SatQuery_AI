"use client";

/* eslint-disable @next/next/no-img-element -- local analysis evidence is served by FastAPI */

import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, Check, LoaderCircle } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { SectionPage } from "@/components/shell/section-page";
import { assetUrl, fetchAnalysis } from "@/lib/api";

export default function HistoryDetailPage() {
  const params = useParams<{ id: string }>();
  const analysis = useQuery({ queryKey: ["analysis", params.id], queryFn: () => fetchAnalysis(params.id) });
  return (
    <SectionPage active="history" eyebrow="ANALYSIS RECORD" title="Evidence record" description="A replay-safe view of the stored result, statistics, confidence and observable execution facts.">
      <Link className="back-link" href="/history"><ArrowLeft size={14} /> Back to history</Link>
      {analysis.isLoading ? <div className="page-empty"><LoaderCircle className="spin" /> Loading analysis…</div> : analysis.isError || !analysis.data ? <div className="page-empty error">This local analysis could not be found.</div> : (
        <div className="analysis-detail">
          <header><div><span className="eyebrow">{analysis.data.task.replaceAll("_", " ")}</span><h2>{analysis.data.query}</h2><p>{analysis.data.answer}</p></div><div className="detail-confidence"><b>{Math.round(analysis.data.confidence.overall * 100)}%</b><span>{analysis.data.confidence.type} confidence</span></div></header>
          {analysis.data.development_label && <div className="detail-warning">{analysis.data.development_label}</div>}
          <div className="detail-grid">
            <section><span className="eyebrow">VISUAL EVIDENCE</span><div className="detail-evidence">{analysis.data.evidence.map((item) => <figure key={item.id}><img src={assetUrl(item.asset_url) ?? ""} alt={item.label} /><figcaption><b>{item.label}</b><span>{item.description}</span></figcaption></figure>)}</div></section>
            <aside><span className="eyebrow">EXECUTION</span>{analysis.data.execution_trace.steps.map((step) => <div className="detail-step" key={step.name}><Check size={11} /><span><b>{step.name}</b><small>{step.detail}</small></span><time>{step.runtime_ms} ms</time></div>)}</aside>
          </div>
        </div>
      )}
    </SectionPage>
  );
}

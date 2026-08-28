import { Beaker, CircleDashed } from "lucide-react";
import { SectionPage } from "@/components/shell/section-page";

const benchmarks = [
  ["Single Image VQA", "VRSBench · RSVQA", "Accuracy · exact match"],
  ["Change Detection", "LEVIR-CD", "IoU · F1 · precision · recall"],
  ["Change VQA", "CDVQA", "Accuracy · exact match"],
  ["Visual Grounding", "VRSBench", "IoU · precision · recall"],
];

export default function BenchmarksPage() {
  return (
    <SectionPage active="benchmarks" eyebrow="MEASURABLE EVALUATION" title="Benchmarks" description="This page only displays metrics produced by local evaluation result files. Placeholder scores are intentionally prohibited.">
      <div className="benchmark-note"><Beaker size={18} /><div><b>Not evaluated yet.</b><span>Run the task-specific scripts in <code>ml/evaluation</code>; generated JSON results can then be surfaced here.</span></div></div>
      <div className="benchmark-grid">
        {benchmarks.map(([name, dataset, metrics]) => <article key={name}><header><CircleDashed size={17} /><span>AWAITING RUN</span></header><h2>{name}</h2><p>{dataset}</p><small>{metrics}</small><div className="metric-placeholder"><span>—</span><b>No measured result</b></div></article>)}
      </div>
    </SectionPage>
  );
}

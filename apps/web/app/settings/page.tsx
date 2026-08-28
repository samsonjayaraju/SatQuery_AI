"use client";

import { useQuery } from "@tanstack/react-query";
import { Cpu, Database, HardDrive, ShieldCheck } from "lucide-react";
import { SectionPage } from "@/components/shell/section-page";
import { fetchHealth } from "@/lib/api";

export default function SettingsPage() {
  const health = useQuery({ queryKey: ["health"], queryFn: fetchHealth });
  return (
    <SectionPage active="settings" eyebrow="LOCAL RUNTIME" title="Settings" description="Runtime configuration comes from the local .env file. No secrets or cloud credentials are required.">
      <div className="settings-grid">
        <article><Cpu /><span>Compute device</span><b>{health.data?.device.toUpperCase() ?? "Unavailable"}</b><small>Automatic priority: CUDA → Apple MPS → CPU</small></article>
        <article><Database /><span>Model mode</span><b>{health.data?.mock_mode ? "Development mock" : "Checkpoint inference"}</b><small>Mock results remain visibly labeled in every analysis.</small></article>
        <article><HardDrive /><span>Data directory</span><b>Local filesystem</b><small>{health.data?.paths.data ?? "Connect the API to resolve the path."}</small></article>
        <article><ShieldCheck /><span>Privacy boundary</span><b>Localhost only</b><small>No mandatory APIs, accounts, telemetry or external storage.</small></article>
      </div>
      <section className="config-table"><span className="eyebrow">ACTIVE DEFAULTS</span><dl><div><dt>Tile size</dt><dd>512 × 512 px</dd></div><div><dt>Tile overlap</dt><dd>64 px</dd></div><div><dt>Maximum upload</dt><dd>500 MB per file</dd></div><div><dt>Model unloading</dt><dd>After each request</dd></div><div><dt>API origin</dt><dd>127.0.0.1:8000</dd></div></dl></section>
    </SectionPage>
  );
}

"use client";

import { useQuery } from "@tanstack/react-query";
import { ArrowUpRight, Clock3, LoaderCircle } from "lucide-react";
import Link from "next/link";
import { SectionPage } from "@/components/shell/section-page";
import { fetchHistory } from "@/lib/api";

export default function HistoryPage() {
  const history = useQuery({ queryKey: ["history"], queryFn: fetchHistory });
  return (
    <SectionPage active="history" eyebrow="LOCAL ANALYSIS LOG" title="History" description="Every completed run is stored as a lightweight, local audit record. Source rasters are not duplicated into history.">
      {history.isLoading ? <div className="page-empty"><LoaderCircle className="spin" /> Reading local history…</div> : history.isError ? <div className="page-empty error">The API is offline. Start the FastAPI service to read history.</div> : history.data?.length ? (
        <div className="history-list">
          {history.data.map((item) => (
            <article key={item.analysis_id}>
              <div className="history-icon"><Clock3 size={17} /></div>
              <div><span>{new Date(item.created_at).toLocaleString()}</span><h2>{item.task.replaceAll("_", " ")}</h2><p>{item.query}</p></div>
              <div className="history-answer"><p>{item.answer}</p><span>{Math.round(item.confidence * 100)}% confidence</span></div>
              <Link href={`/history/${item.analysis_id}`} aria-label={`Open analysis ${item.analysis_id}`}><ArrowUpRight size={16} /></Link>
            </article>
          ))}
        </div>
      ) : <div className="page-empty"><Clock3 /> No analyses yet. Complete a workspace query to create the first local record.</div>}
    </SectionPage>
  );
}

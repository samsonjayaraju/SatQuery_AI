"use client";

/* eslint-disable @next/next/no-img-element -- local object/analysis URLs are intentionally unoptimized */

import dynamic from "next/dynamic";
import { Layers3, Map, Maximize2, ScanSearch, SplitSquareHorizontal } from "lucide-react";
import type { AnalysisResponse, InspectionResponse } from "@/lib/types";
import { assetUrl } from "@/lib/api";

const LeafletCanvas = dynamic(() => import("./leaflet-canvas"), { ssr: false });

export function ViewerPanel({
  inspection,
  result,
  activeEvidence,
  overlayOpacity,
  activeImage,
  split,
  resetKey,
  onOpacity,
  onImage,
  onSplit,
  onFit,
}: {
  inspection: InspectionResponse | null;
  result: AnalysisResponse | null;
  activeEvidence: number;
  overlayOpacity: number;
  activeImage: number;
  split: boolean;
  resetKey: number;
  onOpacity: (opacity: number) => void;
  onImage: (index: number) => void;
  onSplit: () => void;
  onFit: () => void;
}) {
  const current = result?.inspection ?? inspection;
  const images = current?.images ?? [];
  const base = images[activeImage] ?? images[0];
  const baseUrl = assetUrl(base?.thumbnail_url);
  const evidence = result?.evidence[activeEvidence];
  const evidenceUrl = assetUrl(evidence?.asset_url);
  const crs = base?.crs ?? "PIXEL SPACE";

  return (
    <section className="viewer-panel">
      <div className="viewer-toolbar">
        <div><span className="status-dot" /> VIEWER / {base ? base.filename.toUpperCase() : "NO SCENE"}</div>
        <div>
          {images.length > 1 && <button type="button" className={split ? "active" : ""} onClick={onSplit}><SplitSquareHorizontal size={15} /> Split</button>}
          <button type="button" onClick={onFit}><Maximize2 size={14} /> Fit</button>
          <button type="button"><Layers3 size={15} /> {result ? `${result.evidence.length} layers` : "Layers"}</button>
        </div>
      </div>
      {!baseUrl ? (
        <div className="viewer-empty">
          <div className="orbit"><span /><span /><span /><i><Map size={30} /></i></div>
          <span className="eyebrow">GEOSPATIAL CANVAS</span>
          <h2>Your evidence starts here.</h2>
          <p>Load a remote-sensing raster to inspect it, run specialist analysis and verify every answer spatially.</p>
          <div className="coordinate-readout">00° 00′ 00″ N&nbsp;&nbsp; / &nbsp;&nbsp;00° 00′ 00″ E</div>
        </div>
      ) : split && images.length > 1 ? (
        <div className="split-view">
          {images.slice(0, 2).map((image, index) => (
            <figure key={image.filename}><img src={assetUrl(image.thumbnail_url) ?? ""} alt={image.filename} /><figcaption>{index === 0 ? "T1 / OPTICAL" : "T2 / SAR"} · {image.filename}</figcaption></figure>
          ))}
        </div>
      ) : (
        <div className="map-wrap">
          <LeafletCanvas baseUrl={baseUrl} evidenceUrl={evidenceUrl} opacity={overlayOpacity} width={base.width} height={base.height} resetKey={resetKey} />
          <div className="layer-tabs">
            {images.map((image, index) => <button className={activeImage === index ? "active" : ""} key={image.filename} type="button" onClick={() => onImage(index)}>{index === 0 ? "INPUT 1" : "INPUT 2"}</button>)}
            {result && <span><ScanSearch size={13} /> {evidence?.label ?? "Evidence"}</span>}
          </div>
          {result && <label className="opacity-control"><span>Overlay</span><input type="range" min="0" max="100" value={Math.round(overlayOpacity * 100)} onChange={(event) => onOpacity(Number(event.target.value) / 100)} /><b>{Math.round(overlayOpacity * 100)}%</b></label>}
        </div>
      )}
      <div className="viewer-status"><div><span>{crs}</span><span>{base ? `${base.width} × ${base.height}` : "ZOOM 1.0×"}</span><span>{evidence ? evidence.type.toUpperCase() : "NO OVERLAY"}</span></div><span>{base?.georeferenced ? "GEOREFERENCED" : "LOCAL PIXEL COORDINATES"}</span></div>
    </section>
  );
}

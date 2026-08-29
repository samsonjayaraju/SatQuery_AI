"use client";

import { Activity, AlertTriangle, Check, CircleDot, Clock3, Database, FileImage, Layers3, UploadCloud, X } from "lucide-react";
import { useDropzone } from "react-dropzone";
import type { AnalysisStatus, InputMode, InspectionResponse } from "@/lib/types";

const modes: { id: InputMode; label: string; detail: string; icon: typeof FileImage }[] = [
  { id: "single", label: "Single", detail: "One scene", icon: FileImage },
  { id: "bi_temporal", label: "Change", detail: "T1 + T2", icon: Clock3 },
  { id: "cross_modal", label: "Fusion", detail: "Optical + SAR", icon: Layers3 },
];

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function statusText(value: boolean | null) {
  if (value === null) return "N/A";
  return value ? "MATCH" : "REVIEW";
}

export function FileInputPanel({
  mode,
  files,
  inspection,
  status,
  error,
  onModeChange,
  onFiles,
}: {
  mode: InputMode;
  files: File[];
  inspection: InspectionResponse | null;
  status: AnalysisStatus;
  error: string | null;
  onModeChange: (mode: InputMode) => void;
  onFiles: (files: File[]) => void;
}) {
  const expected = mode === "single" ? 1 : 2;
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { "image/tiff": [".tif", ".tiff"], "image/png": [".png"], "image/jpeg": [".jpg", ".jpeg"] },
    maxFiles: expected,
    multiple: expected > 1,
    onDropAccepted: (accepted) => {
      const selected = expected === 2 && accepted.length === 1 && files.length === 1
        ? [files[0], accepted[0]]
        : accepted;
      onFiles(selected.slice(0, expected));
    },
  });

  const labels = mode === "bi_temporal" ? ["T1 · BEFORE", "T2 · AFTER"] : mode === "cross_modal" ? ["OPTICAL", "SAR"] : ["PRIMARY RASTER"];

  return (
    <aside className="input-panel">
      <div className="panel-heading">
        <div><span className="eyebrow">01 · INPUTS</span><h1>Analysis source</h1></div>
        <Database size={17} />
      </div>
      <div className="mode-switch" role="group" aria-label="Analysis mode">
        {modes.map(({ id, label, detail, icon: Icon }) => (
          <button className={mode === id ? "selected" : ""} key={id} type="button" onClick={() => onModeChange(id)}>
            <Icon size={16} /><span>{label}<small>{detail}</small></span>
          </button>
        ))}
      </div>

      <div {...getRootProps({ className: `dropzone ${isDragActive ? "dragging" : ""}` })}>
        <input {...getInputProps()} />
        <span className="drop-icon"><UploadCloud size={22} /></span>
        <strong>{isDragActive ? "Release to inspect" : `Drop ${expected === 1 ? "a satellite image" : "the image pair"}`}</strong>
        <span>GeoTIFF, TIFF, PNG or JPEG</span>
        <small>{expected} FILE{expected > 1 ? "S" : ""} · UP TO 500 MB EACH</small>
      </div>

      {files.length > 0 && (
        <div className="file-list" aria-label="Selected files">
          {files.map((file, index) => {
            const metadata = inspection?.images[index];
            return (
              <div className="file-row" key={`${file.name}-${index}`}>
                <span className="file-type">{labels[index]}</span>
                <div><strong title={file.name}>{file.name}</strong><small>{metadata ? `${metadata.width} × ${metadata.height} · ${metadata.band_count} band${metadata.band_count === 1 ? "" : "s"}` : formatBytes(file.size)}</small></div>
                {metadata ? <Check size={15} className="valid-icon" /> : status === "validating" ? <Activity size={15} className="spin" /> : <button onClick={() => onFiles([])} type="button" aria-label={`Remove ${file.name}`}><X size={14} /></button>}
              </div>
            );
          })}
        </div>
      )}

      {error && <div className="inline-error"><AlertTriangle size={15} /><span>{error}</span></div>}

      {inspection ? (
        <div className="metadata-block">
          <div className="section-label">
            <span className="eyebrow">IMAGE METADATA</span>
            <span className={`validation-badge ${inspection.valid ? "" : "blocked"}`}>
              {inspection.valid ? <Check size={11} /> : <AlertTriangle size={11} />}
              {inspection.valid ? "VALID" : "SOURCE NEEDED"}
            </span>
          </div>
          {inspection.warnings.length > 0 && (
            <div className="input-warning" role="alert">
              <AlertTriangle size={14} />
              <span><b>{inspection.valid ? "Input advisory" : "Analysis paused"}</b>{inspection.warnings[0]}</span>
            </div>
          )}
          {inspection.images.map((image, index) => (
            <dl key={image.filename}>
              {inspection.images.length > 1 && <div className="metadata-title">{labels[index]}</div>}
              <div><dt>Format</dt><dd>{image.format}</dd></div>
              <div><dt>Modality</dt><dd>{image.modality}</dd></div>
              <div><dt>CRS</dt><dd>{image.crs ?? "Not embedded"}</dd></div>
              <div><dt>Resolution</dt><dd>{image.pixel_resolution ? `${image.pixel_resolution.map((v) => v.toFixed(2)).join(" × ")}` : "Pixel space"}</dd></div>
              <div><dt>Data type</dt><dd>{image.data_type}</dd></div>
            </dl>
          ))}
          {inspection.images.length === 2 && (
            <>
              {!inspection.images.every((image) => image.georeferenced) && (
                <div className="pixel-space-warning" role="alert">
                  <AlertTriangle size={14} /><span><b>PIXEL-SPACE ANALYSIS</b>These images contain no shared geographic metadata. Results use image-space alignment and are not geographic area measurements.</span>
                </div>
              )}
              <div className="compatibility-grid" aria-label="Pair compatibility">
                <div><span>CRS match</span><b>{statusText(inspection.compatibility.crs_match)}</b></div>
                <div><span>Spatial overlap</span><b>{inspection.compatibility.overlap === null ? "N/A" : `${Math.round(inspection.compatibility.overlap * 100)}%`}</b></div>
                <div><span>Resolution match</span><b>{statusText(inspection.compatibility.resolution_compatible)}</b></div>
                <div><span>Registration</span><b>{Math.round(inspection.registration.confidence * 100)}%</b></div>
              </div>
              <div className={`compatibility ${inspection.registration.status === "accepted" ? "pass" : "warn"}`}>
                {inspection.registration.status === "accepted" ? <Check size={13} /> : <AlertTriangle size={13} />}
                <span><b>{inspection.registration.status === "accepted" ? "Pair compatible" : "Review alignment"}</b>{inspection.registration.method.replaceAll("_", " ")}</span>
              </div>
            </>
          )}
        </div>
      ) : (
        <div className="empty-metadata">
          <span className="eyebrow">IMAGE METADATA</span>
          <div className="metadata-illustration"><Activity size={20} /><span /></div>
          <p>Raster dimensions, CRS, bands and pixel resolution will appear after validation.</p>
        </div>
      )}
      <div className="privacy-note"><CircleDot size={14} /><span><b>Local by design</b>Imagery never leaves this machine.</span></div>
    </aside>
  );
}

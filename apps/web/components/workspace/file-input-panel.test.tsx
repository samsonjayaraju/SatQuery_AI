import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { InspectionResponse } from "@/lib/types";
import { FileInputPanel } from "./file-input-panel";


const blockedInspection: InspectionResponse = {
  valid: false,
  input_mode: "single",
  images: [{
    filename: "paper-figure.png",
    file_size_bytes: 1024,
    width: 420,
    height: 420,
    band_count: 3,
    data_type: "RGB",
    crs: null,
    transform: null,
    bounds: null,
    pixel_resolution: null,
    nodata: null,
    georeferenced: false,
    modality: "optical",
    format: "PNG",
    thumbnail_url: null,
    display_width: null,
    display_height: null,
  }],
  compatibility: {
    crs_match: null,
    overlap: null,
    co_registered: null,
    dimensions_match: null,
    resolution_compatible: null,
    warnings: [],
  },
  visual_quality: {
    status: "unsupported",
    score: 0.1,
    flags: ["composite_figure"],
    recommendation: "Upload the original satellite panel separately.",
  },
  registration: {
    method: "not_required",
    confidence: 1,
    status: "not_required",
    transform: null,
    warnings: [],
  },
  warnings: ["Upload the original satellite panel separately."],
};


describe("FileInputPanel", () => {
  it("shows a blocking source warning for composite figures", () => {
    render(
      <FileInputPanel
        mode="single"
        files={[]}
        inspection={blockedInspection}
        status="ready"
        error={null}
        onModeChange={vi.fn()}
        onFiles={vi.fn()}
      />,
    );

    expect(screen.getByText("SOURCE NEEDED")).toBeInTheDocument();
    expect(screen.getByText("Analysis paused")).toBeInTheDocument();
    expect(screen.getByText("Upload the original satellite panel separately.")).toBeInTheDocument();
  });
});

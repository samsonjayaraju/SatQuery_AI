import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AssistantPanel } from "./assistant-panel";


const callbacks = {
  onQuery: vi.fn(),
  onAnalyze: vi.fn(),
  onEvidence: vi.fn(),
  onReport: vi.fn(),
};

afterEach(cleanup);

describe("AssistantPanel", () => {
  it("shows real queued-job progress and disables duplicate submission", () => {
    render(
      <AssistantPanel
        mode="single"
        status="processing"
        job={{
          job_id: "job-1",
          status: "processing",
          message: "Running tiled inference",
          progress: 0.52,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          analysis_id: null,
          result: null,
          error_code: null,
        }}
        query="Describe the scene"
        result={null}
        activeEvidence={0}
        canAnalyze
        reportBusy={false}
        {...callbacks}
      />,
    );

    expect(screen.getByText("Running tiled inference")).toBeInTheDocument();
    expect(screen.getByText(/52%/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /analyze/i })).toBeDisabled();
    expect(screen.getByLabelText("Analysis question")).toBeDisabled();
  });

  it("lets a ready user choose an analysis prompt", () => {
    callbacks.onQuery.mockClear();
    render(
      <AssistantPanel
        mode="bi_temporal"
        status="ready"
        job={null}
        query=""
        result={null}
        activeEvidence={0}
        canAnalyze
        reportBusy={false}
        {...callbacks}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /what changed between these two dates/i }));
    expect(callbacks.onQuery).toHaveBeenCalledWith("What changed between these two dates?");
  });

  it("keeps the query field reachable before imagery is loaded", () => {
    render(
      <AssistantPanel
        mode="single"
        status="empty"
        job={null}
        query=""
        result={null}
        activeEvidence={0}
        canAnalyze={false}
        reportBusy={false}
        {...callbacks}
      />,
    );

    expect(screen.getByLabelText("Analysis question")).toBeEnabled();
    expect(screen.getByRole("button", { name: /analyze/i })).toBeDisabled();
  });

  it("allows typing a Fusion query while the image pair is validating", () => {
    callbacks.onQuery.mockClear();
    render(
      <AssistantPanel
        mode="cross_modal"
        status="validating"
        job={null}
        query=""
        result={null}
        activeEvidence={0}
        canAnalyze={false}
        reportBusy={false}
        {...callbacks}
      />,
    );

    const queryField = screen.getByLabelText("Analysis question");
    expect(queryField).toBeEnabled();
    fireEvent.change(queryField, { target: { value: "Where do optical and SAR evidence agree?" } });
    expect(callbacks.onQuery).toHaveBeenCalledWith("Where do optical and SAR evidence agree?");
    expect(screen.getByRole("button", { name: /analyze/i })).toBeDisabled();
  });
});

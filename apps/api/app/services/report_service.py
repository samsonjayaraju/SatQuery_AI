from __future__ import annotations

from html import escape
from pathlib import Path

from app.api.schemas.analysis import AnalysisResponse


class ReportService:
    def __init__(self, data_dir: Path):
        self.directory = data_dir.resolve() / "reports"
        self.directory.mkdir(parents=True, exist_ok=True)

    def build_html(self, result: AnalysisResponse) -> Path:
        rows = "".join(
            f"<tr><th>{escape(str(key).replace('_', ' ').title())}</th><td>{escape(str(value))}</td></tr>"
            for key, value in result.statistics.items()
        )
        model_list = ", ".join(result.execution_trace.models) or "No learned model invoked"
        tool_list = ", ".join(result.execution_trace.tools)
        images = "".join(
            f'<figure><img src="{escape(item.asset_url.replace("/assets/", "../"))}" alt="{escape(item.label)}"><figcaption>{escape(item.description)}</figcaption></figure>'
            for item in result.evidence
            if item.asset_url
        )
        document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>SatQuery AI Report {escape(result.analysis_id)}</title>
<style>body{{font:15px/1.55 Arial,sans-serif;color:#17222a;max-width:960px;margin:40px auto;padding:0 28px}}header{{border-bottom:3px solid #0f766e;padding-bottom:18px}}h1{{margin:0}}.eyebrow{{color:#0f766e;letter-spacing:.12em;text-transform:uppercase;font-weight:700}}.answer{{font-size:20px;background:#eef7f5;padding:20px;border-left:4px solid #0f766e}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #d8e0e3}}th{{width:36%}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}img{{max-width:100%;border-radius:6px}}small{{color:#60717a}}@media print{{body{{margin:0}}}}</style></head>
<body><header><div class="eyebrow">SatQuery AI · Analysis Report</div><h1>{escape(result.task.replace('_', ' ').title())}</h1>
<p>ID {escape(result.analysis_id)} · {escape(result.created_at.isoformat())}</p></header>
<h2>Query</h2><p>{escape(result.query)}</p><h2>Result</h2><p class="answer">{escape(result.answer)}</p>
<p><strong>Confidence:</strong> {result.confidence.overall:.0%} ({escape(result.confidence.type)})<br><small>{escape(result.confidence.note)}</small></p>
<h2>Statistics</h2><table>{rows}</table><h2>Visual evidence</h2><div class="grid">{images}</div>
<h2>Execution summary</h2><p><strong>Models:</strong> {escape(model_list)}<br><strong>Tools:</strong> {escape(tool_list)}<br><strong>Runtime:</strong> {result.runtime_ms} ms</p>
<h2>Inputs</h2><p>{escape(', '.join(image.filename for image in result.inspection.images))}</p>
<h2>Limitations</h2><p>This research prototype uses heuristic confidence unless a calibrated checkpoint is available. Results can be affected by cloud, resolution, georegistration, sensor differences and domain shift. Verify high-impact decisions with authoritative geospatial data and expert review.</p>
</body></html>"""
        path = self.directory / f"satquery-{result.analysis_id}.html"
        path.write_text(document, encoding="utf-8")
        return path

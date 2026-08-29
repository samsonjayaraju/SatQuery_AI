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
        confidence_rows = "".join(
            f"<tr><th>{escape(key.replace('_', ' ').title())}</th><td>{value:.1%}</td></tr>"
            for key, value in result.confidence.components.items()
        )
        transition_rows = "".join(
            "<tr>"
            f"<td>{escape('Unchanged' if item.from_class == 'unchanged' else item.from_class.replace('_', ' ').title())}</td>"
            f"<td>{escape('—' if item.to_class == 'unchanged' else item.to_class.replace('_', ' ').title())}</td>"
            f"<td>{item.percent:.2f}%</td><td>{item.pixel_count:,}</td>"
            f"<td>{'—' if item.area_hectares is None else f'{item.area_hectares:,.4f} ha'}</td></tr>"
            for item in result.transitions
        ) or '<tr><td colspan="5">Not applicable to this analysis mode.</td></tr>'
        metadata_rows = "".join(
            "<tr>"
            f"<th>{escape(image.filename)}</th>"
            f"<td>{image.width} × {image.height}</td><td>{image.band_count}</td>"
            f"<td>{escape(image.crs or 'Pixel space')}</td><td>{escape(image.modality)}</td></tr>"
            for image in result.inspection.images
        )
        compatibility = result.inspection.compatibility
        compatibility_rows = "".join(
            f"<tr><th>{escape(key.replace('_', ' ').title())}</th><td>{escape(str(value))}</td></tr>"
            for key, value in {
                "CRS match": compatibility.crs_match,
                "Spatial overlap": compatibility.overlap,
                "Resolution compatible": compatibility.resolution_compatible,
                "Registration method": result.inspection.registration.method,
                "Registration confidence": f"{result.inspection.registration.confidence:.1%}",
            }.items()
        )
        parameter_rows = "".join(
            f"<tr><th>{escape(str(key).replace('_', ' ').title())}</th><td>{escape(str(value))}</td></tr>"
            for key, value in result.execution_trace.parameters.items()
        )
        trace_rows = "".join(
            f"<tr><th>{escape(step.name)}</th><td>{escape(step.detail)}</td><td>{step.runtime_ms} ms</td></tr>"
            for step in result.execution_trace.steps
        )
        warning_items = "".join(f"<li>{escape(warning)}</li>" for warning in result.warnings)
        if len(result.inspection.images) > 1 and not all(image.georeferenced for image in result.inspection.images):
            pixel_warning = (
                "<div class=\"warning\"><strong>PIXEL-SPACE ANALYSIS</strong><br>"
                "These images contain no shared geographic metadata. Results are based on image-space alignment "
                "and must not be interpreted as geographic area measurements.</div>"
            )
        else:
            pixel_warning = ""
        images = "".join(
            f'<figure><img src="{escape(item.asset_url.replace("/assets/", "../"))}" alt="{escape(item.label)}"><figcaption>{escape(item.description)}</figcaption></figure>'
            for item in result.evidence
            if item.asset_url
        )
        document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>SatQuery AI Report {escape(result.analysis_id)}</title>
<style>body{{font:15px/1.55 Arial,sans-serif;color:#17222a;max-width:960px;margin:40px auto;padding:0 28px}}header{{border-bottom:3px solid #0f766e;padding-bottom:18px}}h1{{margin:0}}h2{{margin-top:30px}}.eyebrow{{color:#0f766e;letter-spacing:.12em;text-transform:uppercase;font-weight:700}}.answer{{font-size:20px;background:#eef7f5;padding:20px;border-left:4px solid #0f766e}}.warning{{margin:18px 0;padding:14px;color:#744b00;border:1px solid #d9a643;background:#fff7df}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #d8e0e3;vertical-align:top}}th{{width:32%}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}figure{{margin:0}}img{{max-width:100%;border-radius:6px}}small{{color:#60717a}}@media print{{body{{margin:0}}}}</style></head>
<body><header><div class="eyebrow">SatQuery AI · Analysis Report</div><h1>{escape(result.task.replace('_', ' ').title())}</h1>
<p>ID {escape(result.analysis_id)} · {escape(result.created_at.isoformat())}</p></header>
{pixel_warning}<h2>Query and task</h2><p><strong>Mode:</strong> {escape(result.inspection.input_mode.replace('_', ' '))}<br><strong>Detected task:</strong> {escape(result.task)}<br><strong>Query:</strong> {escape(result.query)}</p>
<h2>Result</h2><p class="answer">{escape(result.answer)}</p>
<h2>Confidence</h2><p><strong>Overall:</strong> {result.confidence.overall:.0%} ({escape(result.confidence.type)})<br><small>{escape(result.confidence.note)}</small></p><table>{confidence_rows}</table>
<h2>Measurements</h2><table>{rows}</table>
<h2>Land-cover transitions</h2><table><thead><tr><th>From</th><th>To</th><th>Scene share</th><th>Pixels</th><th>Area</th></tr></thead><tbody>{transition_rows}</tbody></table>
<h2>Visual evidence</h2><div class="grid">{images}</div>
<h2>Inputs and metadata</h2><table><thead><tr><th>File</th><th>Dimensions</th><th>Bands</th><th>CRS</th><th>Modality</th></tr></thead><tbody>{metadata_rows}</tbody></table>
<h2>Input compatibility</h2><table>{compatibility_rows}</table>
<h2>Execution summary</h2><p><strong>Models / baselines:</strong> {escape(model_list)}<br><strong>Tools:</strong> {escape(tool_list)}<br><strong>Total runtime:</strong> {result.runtime_ms} ms</p><table>{parameter_rows}</table><table>{trace_rows}</table>
<h2>Warnings</h2>{f'<ul>{warning_items}</ul>' if warning_items else '<p>No additional warnings.</p>'}
<h2>Limitations</h2><p>This research prototype uses heuristic confidence unless a calibrated checkpoint is available. Results can be affected by cloud, resolution, georegistration, sensor differences and domain shift. Verify high-impact decisions with authoritative geospatial data and expert review.</p>
</body></html>"""
        path = self.directory / f"satquery-{result.analysis_id}.html"
        path.write_text(document, encoding="utf-8")
        return path

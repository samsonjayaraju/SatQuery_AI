from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.agent.query_interpreter import QueryInterpreter
from app.api.schemas.analysis import (
    AnalysisResponse,
    EvidenceItem,
    ExecutionTrace,
    InspectionResponse,
    TraceStep,
)
from app.core.config import Settings
from app.reasoning.change_reasoner import explain_change
from app.reasoning.confidence_engine import ConfidenceEngine
from app.registry.model_registry import ModelRegistry
from app.registry.tool_registry import ToolRegistry
from app.remote_sensing.input_inspector import inspect_inputs
from app.remote_sensing.preprocessing import (
    load_visual,
    optical_probabilities,
    resize_like,
    sar_probabilities,
)
from app.remote_sensing.visualization import bounding_box, draw_box, heatmap, overlay_mask, save_image
from app.services.history_service import HistoryService

logger = logging.getLogger(__name__)
CLASS_THRESHOLDS = {"water": 0.58, "vegetation": 0.58, "built_up": 0.45, "bare_land": 0.5, "agriculture": 0.58}


class AnalysisService:
    def __init__(
        self,
        settings: Settings,
        model_registry: ModelRegistry,
        tool_registry: ToolRegistry,
        history: HistoryService,
    ):
        self.settings = settings
        self.model_registry = model_registry
        self.tool_registry = tool_registry
        self.history = history
        self.interpreter = QueryInterpreter()
        self.confidence = ConfidenceEngine()

    def _step(self, name: str, kind: str, started: float, detail: str) -> TraceStep:
        return TraceStep(
            name=name,
            kind=kind,
            status="completed",
            runtime_ms=max(1, round((time.perf_counter() - started) * 1000)),
            detail=detail,
        )

    def _asset_url(self, analysis_id: str, name: str) -> str:
        return f"/assets/outputs/{analysis_id}/{name}"

    def _prepare(self, paths: list[Path], analysis_id: str) -> tuple[list[np.ndarray], list[str]]:
        output = self.settings.data_dir.resolve() / "outputs" / analysis_id
        output.mkdir(parents=True, exist_ok=True)
        arrays: list[np.ndarray] = []
        urls: list[str] = []
        for index, path in enumerate(paths):
            array = load_visual(path)
            max_side = max(array.shape[:2])
            if max_side > 1024:
                scale = 1024 / max_side
                array = load_visual(path, (max(1, round(array.shape[1] * scale)), max(1, round(array.shape[0] * scale))))
            name = f"input-{index + 1}.png"
            save_image(array, output / name)
            arrays.append(array)
            urls.append(self._asset_url(analysis_id, name))
        return arrays, urls

    @staticmethod
    def _coverage(probability: np.ndarray, threshold: float = 0.58) -> float:
        return round(float((probability >= threshold).mean() * 100), 2)

    def analyze(self, paths: list[Path], query: str, mode: str) -> AnalysisResponse:
        total_started = time.perf_counter()
        analysis_id = str(uuid.uuid4())
        steps: list[TraceStep] = []

        started = time.perf_counter()
        arrays, urls = self._prepare(paths, analysis_id)
        inspection: InspectionResponse = inspect_inputs(paths, mode, urls)
        steps.append(self._step("Input inspection", "tool", started, "Validated raster structure, metadata and pair compatibility."))

        started = time.perf_counter()
        intent = self.interpreter.classify(query, mode)
        steps.append(self._step("Query interpretation", "system", started, f"Detected {intent.intent} at {intent.confidence:.0%} routing confidence."))

        started = time.perf_counter()
        if mode == "bi_temporal":
            answer, stats, evidence, confidence, models, tools = self._change(
                arrays[0], arrays[1], analysis_id, intent.entities.get("target_class")
            )
        elif mode == "cross_modal":
            answer, stats, evidence, confidence, models, tools = self._cross_modal(
                arrays[0], arrays[1], analysis_id, intent.entities.get("target_class")
            )
        else:
            answer, stats, evidence, confidence, models, tools = self._single(
                arrays[0], analysis_id, intent.intent, intent.entities.get("target_class")
            )
        steps.append(self._step("Specialist workflow", "model", started, f"Executed {', '.join(models)} with spatial evidence generation."))

        runtime_ms = max(1, round((time.perf_counter() - total_started) * 1000))
        steps.append(TraceStep(name="Evidence integration", kind="system", status="completed", runtime_ms=1, detail="Integrated spatial outputs, statistics and heuristic confidence."))
        trace = ExecutionTrace(
            task=intent.intent,
            input_mode=mode,
            models=models,
            tools=tools,
            parameters={
                "tile_size": self.settings.tile_size,
                "tile_overlap": self.settings.tile_overlap,
                "change_threshold": self.settings.change_threshold,
                "device": self.model_registry.device,
            },
            steps=steps,
            runtime_ms=runtime_ms,
            status="success",
            mock_mode=self.settings.mock_mode,
        )
        result = AnalysisResponse(
            analysis_id=analysis_id,
            created_at=datetime.now(timezone.utc),
            task=intent.intent,
            query=query,
            answer=answer,
            development_label=(
                "Development Mock Result · deterministic local baseline; learned checkpoints were not invoked."
                if self.settings.mock_mode
                else None
            ),
            confidence=confidence,
            evidence=evidence,
            statistics=stats,
            inspection=inspection,
            execution_trace=trace,
            runtime_ms=runtime_ms,
        )
        self.history.save(result)
        logger.info("analysis completed", extra={"analysis_id": analysis_id, "runtime_ms": runtime_ms})
        return result

    def _single(self, image: np.ndarray, analysis_id: str, intent: str, target: str | None):
        probabilities = optical_probabilities(image)
        stats = {f"{label}_percent": self._coverage(value, CLASS_THRESHOLDS.get(label, 0.58)) for label, value in probabilities.items()}
        ranked = sorted(((label, value) for label, value in stats.items()), key=lambda item: float(item[1]), reverse=True)
        primary = target or ranked[0][0].replace("_percent", "")
        probability = probabilities.get(primary, probabilities["built_up"])
        output = self.settings.data_dir.resolve() / "outputs" / analysis_id
        evidence: list[EvidenceItem] = []

        overlay_name = f"{primary}-overlay.png"
        threshold = CLASS_THRESHOLDS.get(primary, 0.58)
        overlay_mask(image, probability, output / overlay_name, primary, threshold)
        evidence.append(
            EvidenceItem(
                id="single-overlay",
                type="overlay",
                label=f"{primary.replace('_', ' ').title()} evidence",
                description="High-scoring candidate pixels from the deterministic spectral baseline.",
                confidence=round(float(probability.mean()), 3),
                asset_url=self._asset_url(analysis_id, overlay_name),
                color="#4fd1c5",
            )
        )
        if intent == "REGION_GROUNDING":
            coordinates = bounding_box(probability, threshold)
            if coordinates:
                box_name = f"{primary}-grounding.png"
                draw_box(image, coordinates, output / box_name, primary)
                evidence.insert(
                    0,
                    EvidenceItem(
                        id="grounding-box",
                        type="bounding_box",
                        label=f"Candidate {primary.replace('_', ' ')} extent",
                        description="Normalized image coordinates for the high-evidence region.",
                        confidence=round(float(probability[probability >= threshold].mean()), 3),
                        asset_url=self._asset_url(analysis_id, box_name),
                        coordinates=coordinates,
                        color="#37b8ff",
                    ),
                )
        dominant = ", ".join(label.replace("_percent", "").replace("_", " ") for label, _ in ranked[:3])
        if intent == "REGION_GROUNDING":
            answer = f"The strongest {primary.replace('_', ' ')} candidate region is highlighted. The box covers the spatial extent of pixels passing the evidence threshold."
        elif intent in {"WATER_ANALYSIS", "VEGETATION_ANALYSIS", "BUILT_UP_ANALYSIS"}:
            answer = f"Approximately {stats[f'{primary}_percent']:.1f}% of the scene passes the {primary.replace('_', ' ')} evidence threshold. Review the overlay before treating this as a land-cover map."
        else:
            answer = f"The scene is primarily characterized by {dominant}. The deterministic baseline estimates {stats['vegetation_percent']:.1f}% vegetation, {stats['built_up_percent']:.1f}% built-up evidence and {stats['water_percent']:.1f}% water evidence."
        confidence = self.confidence.from_probability(probability, threshold)
        models = ["Spectral Land-Cover Baseline v1"]
        if intent in {"SINGLE_IMAGE_VQA", "IMAGE_CAPTION", "REGION_GROUNDING"}:
            models.insert(0, "Mock Remote-Sensing VLM Adapter")
        capabilities = ["preprocessing", "overlay_generation", "statistics", "confidence"]
        if primary == "water": capabilities.append("water_index")
        elif primary == "vegetation": capabilities.append("vegetation_index")
        elif primary == "built_up": capabilities.append("built_up_index")
        tools = self.tool_registry.names_for(capabilities)
        return answer, stats, evidence, confidence, models, tools

    def _change(self, before: np.ndarray, after: np.ndarray, analysis_id: str, target: str | None):
        after = resize_like(after, before)
        before_float = before.astype(np.float32) / 255.0
        after_float = after.astype(np.float32) / 255.0
        raw = np.mean(np.abs(after_float - before_float), axis=2)
        high = max(float(np.percentile(raw, 98)), 0.08)
        probability = np.clip(raw / high, 0, 1)
        before_classes = optical_probabilities(before)
        after_classes = optical_probabilities(after)
        stats: dict[str, float] = {
            "changed_area_percent": self._coverage(probability, self.settings.change_threshold),
            "built_up_before_percent": self._coverage(before_classes["built_up"], CLASS_THRESHOLDS["built_up"]),
            "built_up_after_percent": self._coverage(after_classes["built_up"], CLASS_THRESHOLDS["built_up"]),
            "vegetation_before_percent": self._coverage(before_classes["vegetation"]),
            "vegetation_after_percent": self._coverage(after_classes["vegetation"]),
            "water_before_percent": self._coverage(before_classes["water"]),
            "water_after_percent": self._coverage(after_classes["water"]),
        }
        stats["built_up_change_pp"] = round(stats["built_up_after_percent"] - stats["built_up_before_percent"], 2)
        stats["vegetation_change_pp"] = round(stats["vegetation_after_percent"] - stats["vegetation_before_percent"], 2)
        stats["water_change_pp"] = round(stats["water_after_percent"] - stats["water_before_percent"], 2)
        output = self.settings.data_dir.resolve() / "outputs" / analysis_id
        overlay_name = "change-overlay.png"
        heatmap(probability, output / "change-heatmap.png")
        overlay_mask(after, probability, output / overlay_name, "change", self.settings.change_threshold)
        coordinates = bounding_box(probability, self.settings.change_threshold)
        evidence = [
            EvidenceItem(id="change-overlay", type="overlay", label="Change overlay", description="Pixels with material before/after spectral difference overlaid on T2.", confidence=round(float(probability.mean()), 3), asset_url=self._asset_url(analysis_id, overlay_name), coordinates=coordinates, color="#f55f72"),
            EvidenceItem(id="change-heatmap", type="heatmap", label="Change probability", description="Transparent change-evidence heatmap generated from normalized pixel differences.", confidence=round(float(probability.mean()), 3), asset_url=self._asset_url(analysis_id, "change-heatmap.png"), color="#f55f72"),
        ]
        answer = explain_change(stats, target)
        confidence = self.confidence.from_probability(probability, self.settings.change_threshold, spatial_quality=0.78)
        tools = self.tool_registry.names_for(["change_detection", "overlay_generation", "statistics", "confidence"])
        return answer, stats, evidence, confidence, ["Pixel Change Baseline v1", "Spectral Land-Cover Baseline v1", "Change Reasoner v1"], tools

    def _cross_modal(self, optical: np.ndarray, sar: np.ndarray, analysis_id: str, target: str | None):
        sar = resize_like(sar, optical)
        optical_scores = optical_probabilities(optical)
        sar_scores = sar_probabilities(sar)
        primary = target if target in {"water", "built_up"} else "water"
        optical_probability = optical_scores[primary]
        sar_probability = sar_scores[primary]
        fused = np.clip(optical_probability * 0.46 + sar_probability * 0.54, 0, 1)
        agreement = float(1 - np.mean(np.abs(optical_probability - sar_probability)))
        stats = {
            "target_class": primary,
            "optical_evidence_percent": self._coverage(optical_probability, CLASS_THRESHOLDS[primary]),
            "sar_evidence_percent": self._coverage(sar_probability, CLASS_THRESHOLDS[primary]),
            "fused_evidence_percent": self._coverage(fused, CLASS_THRESHOLDS[primary]),
            "cross_sensor_agreement_percent": round(agreement * 100, 2),
            "optical_mean_probability": round(float(optical_probability.mean()), 3),
            "sar_mean_probability": round(float(sar_probability.mean()), 3),
            "fused_mean_probability": round(float(fused.mean()), 3),
        }
        output = self.settings.data_dir.resolve() / "outputs" / analysis_id
        items = []
        for label, image, probability, color in (
            ("optical", optical, optical_probability, "#37b8ff"),
            ("sar", sar, sar_probability, "#f0b45b"),
            ("fused", optical, fused, "#52e0c4"),
        ):
            name = f"{label}-{primary}-evidence.png"
            overlay_mask(image, probability, output / name, primary if label != "fused" else "fused", CLASS_THRESHOLDS[primary])
            items.append(EvidenceItem(id=f"{label}-evidence", type="overlay", label=f"{label.title()} evidence", description=f"{label.title()} contribution to the {primary.replace('_', ' ')} result.", confidence=round(float(probability.mean()), 3), asset_url=self._asset_url(analysis_id, name), color=color))
        answer = (
            f"Optical and SAR evidence were evaluated independently, then fused for {primary.replace('_', ' ')} analysis. "
            f"The fused result marks {stats['fused_evidence_percent']:.1f}% of the scene, with {stats['cross_sensor_agreement_percent']:.1f}% cross-sensor agreement."
        )
        confidence = self.confidence.from_probability(fused, CLASS_THRESHOLDS[primary], agreement=agreement, spatial_quality=0.8)
        tools = self.tool_registry.names_for(["optical_analysis", "sar_analysis", "satfusion", "overlay_generation", "confidence"])
        return answer, stats, items, confidence, ["Optical Spectral Baseline v1", "SAR Backscatter Baseline v1", "SatFusion Concatenation Baseline v1"], tools

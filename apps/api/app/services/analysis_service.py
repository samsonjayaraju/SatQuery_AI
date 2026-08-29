from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from app.agent.query_interpreter import QueryInterpreter
from app.api.schemas.analysis import (
    AnalysisResponse,
    EvidenceItem,
    ExecutionTrace,
    InspectionResponse,
    TraceStep,
)
from app.core.config import Settings
from app.core.exceptions import SatQueryError
from app.reasoning.change_reasoner import explain_change
from app.reasoning.confidence_engine import ConfidenceEngine
from app.registry.model_registry import ModelRegistry
from app.registry.tool_registry import ToolRegistry
from app.models.remoteclip import RemoteCLIPService
from app.models.changeformer import ChangeFormerService
from app.models.satfusion import SatFusionService
from app.remote_sensing.input_inspector import inspect_inputs
from app.remote_sensing.alignment import align_visual_pair
from app.remote_sensing.preprocessing import (
    load_visual,
    optical_probabilities,
    resize_like,
    sar_probabilities,
)
from app.remote_sensing.visualization import (
    bounding_box,
    draw_box,
    heatmap,
    largest_polygon,
    overlay_mask,
    retain_largest_component,
    save_image,
)
from app.remote_sensing.tiling import tiled_dict_predict
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
        remoteclip: RemoteCLIPService | None = None,
        changeformer: ChangeFormerService | None = None,
        satfusion: SatFusionService | None = None,
    ):
        self.settings = settings
        self.model_registry = model_registry
        self.tool_registry = tool_registry
        self.history = history
        self.interpreter = QueryInterpreter()
        self.confidence = ConfidenceEngine()
        self.remoteclip = remoteclip
        self.changeformer = changeformer
        self.satfusion = satfusion or SatFusionService()

    def release_models(self) -> None:
        if not self.settings.model_unload_after_request:
            return
        if self.remoteclip and self.remoteclip.loaded:
            self.remoteclip.unload()
            self.model_registry.mark_loaded("remoteclip_encoder", False)
            self.model_registry.mark_loaded("satquery_adapter", False)
        if self.changeformer and self.changeformer.loaded:
            self.changeformer.unload()
            self.model_registry.mark_loaded("changeformer", False)

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

    def _prepare(self, paths: list[Path], analysis_id: str) -> tuple[list[np.ndarray], list[str], str]:
        output = self.settings.data_dir.resolve() / "outputs" / analysis_id
        output.mkdir(parents=True, exist_ok=True)
        arrays: list[np.ndarray] = []
        urls: list[str] = []
        if len(paths) == 2:
            alignment = align_visual_pair(paths[0], paths[1])
            source_arrays = [alignment.first, alignment.second]
            alignment_method = alignment.method
        else:
            source_arrays = [load_visual(paths[0])]
            alignment_method = "not_required"
        for index, array in enumerate(source_arrays):
            max_side = max(array.shape[:2])
            preview_array = array
            if max_side > 1024:
                scale = 1024 / max_side
                preview_array = np.asarray(
                    Image.fromarray(array).resize(
                        (max(1, round(array.shape[1] * scale)), max(1, round(array.shape[0] * scale))),
                        Image.Resampling.BILINEAR,
                    )
                )
            name = f"input-{index + 1}.png"
            save_image(preview_array, output / name)
            arrays.append(array)
            urls.append(self._asset_url(analysis_id, name))
        return arrays, urls, alignment_method

    @staticmethod
    def _coverage(probability: np.ndarray, threshold: float = 0.58) -> float:
        return round(float((probability >= threshold).mean() * 100), 2)

    def analyze(
        self,
        paths: list[Path],
        query: str,
        mode: str,
        progress: Callable[[str, str], None] | None = None,
    ) -> AnalysisResponse:
        total_started = time.perf_counter()
        analysis_id = str(uuid.uuid4())
        steps: list[TraceStep] = []

        started = time.perf_counter()
        if progress:
            progress("validating", "Inspecting raster metadata and compatibility")
        arrays, urls, alignment_method = self._prepare(paths, analysis_id)
        inspection: InspectionResponse = inspect_inputs(paths, mode, urls)
        for metadata, array in zip(inspection.images, arrays):
            metadata.display_width = array.shape[1]
            metadata.display_height = array.shape[0]
        steps.append(self._step("Input inspection", "tool", started, f"Validated raster structure and compatibility; alignment: {alignment_method}."))

        started = time.perf_counter()
        intent = self.interpreter.classify(query, mode)
        steps.append(self._step("Query interpretation", "system", started, f"Detected {intent.intent} at {intent.confidence:.0%} routing confidence."))

        started = time.perf_counter()
        if progress:
            if not self.settings.mock_mode:
                progress("loading_model", "Loading the selected local specialist model")
            progress("processing", f"Executing specialist workflow for {intent.intent}")
        if mode == "bi_temporal":
            answer, stats, evidence, confidence, models, tools, learned = self._change(
                arrays[0], arrays[1], analysis_id, intent.entities.get("target_class")
            )
        elif mode == "cross_modal":
            answer, stats, evidence, confidence, models, tools, learned = self._cross_modal(
                arrays[0], arrays[1], analysis_id, intent.entities.get("target_class")
            )
        else:
            answer, stats, evidence, confidence, models, tools, learned = self._single(
                arrays[0], analysis_id, query, intent.intent, intent.entities.get("target_class")
            )
        steps.append(self._step("Specialist workflow", "model", started, f"Executed {', '.join(models)} with tiled spatial evidence generation."))

        runtime_ms = max(1, round((time.perf_counter() - total_started) * 1000))
        if progress:
            progress("integrating", "Integrating evidence, confidence, statistics and audit trace")
        steps.append(TraceStep(name="Evidence integration", kind="system", status="completed", runtime_ms=1, detail="Integrated spatial outputs, statistics and confidence components."))
        trace = ExecutionTrace(
            task=intent.intent,
            input_mode=mode,
            models=models,
            tools=tools,
            parameters={
                "tile_size": self.settings.tile_size,
                "tile_overlap": self.settings.tile_overlap,
                "alignment": alignment_method,
                "change_threshold": stats.get("change_threshold", self.settings.change_threshold),
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
                if not learned
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

    def _single(self, image: np.ndarray, analysis_id: str, query: str, intent: str, target: str | None):
        learned = bool(not self.settings.mock_mode and self.remoteclip and self.remoteclip.available)
        if not learned and not self.settings.mock_mode and intent in {"SINGLE_IMAGE_VQA", "IMAGE_CAPTION", "REGION_GROUNDING"}:
            raise SatQueryError(
                "MODEL_UNAVAILABLE",
                "RemoteCLIP is required for learned VQA, captioning, and grounding when MOCK_MODE=false.",
                503,
            )
        if learned:
            probabilities, learned_landcover_model = self.remoteclip.landcover_probabilities(image)
            self.model_registry.mark_loaded("remoteclip_encoder")
            if self.remoteclip.adapter_available:
                self.model_registry.mark_loaded("satquery_adapter")
            thresholds = {label: 0.32 for label in probabilities}
        else:
            probabilities = tiled_dict_predict(
                image, optical_probabilities, self.settings.tile_size, self.settings.tile_overlap
            )
            learned_landcover_model = "Spectral Land-Cover Baseline v1"
            thresholds = CLASS_THRESHOLDS
        stats = {f"{label}_percent": self._coverage(value, thresholds.get(label, 0.32)) for label, value in probabilities.items()}
        ranked = sorted(((label, value) for label, value in stats.items()), key=lambda item: float(item[1]), reverse=True)
        primary = target or ranked[0][0].replace("_percent", "")
        probability = probabilities.get(primary, probabilities["built_up"])
        learned_answer = None
        if learned and intent == "REGION_GROUNDING":
            probability, _ = self.remoteclip.ground(image, primary, target)
        elif learned and intent in {"SINGLE_IMAGE_VQA", "IMAGE_CAPTION"}:
            learned_answer = self.remoteclip.answer(image, query, target, caption=intent == "IMAGE_CAPTION")
        output = self.settings.data_dir.resolve() / "outputs" / analysis_id
        evidence: list[EvidenceItem] = []

        overlay_name = f"{primary}-overlay.png"
        threshold = thresholds.get(primary, 0.32)
        if intent == "REGION_GROUNDING":
            if not np.any(probability >= threshold):
                threshold = max(float(probability.max()) * 0.9, 1e-6)
            probability = retain_largest_component(probability, threshold)
        overlay_mask(image, probability, output / overlay_name, primary, threshold)
        evidence.append(
            EvidenceItem(
                id="single-overlay",
                type="overlay",
                label=f"{primary.replace('_', ' ').title()} evidence",
                description=("Learned RemoteCLIP/adapter evidence stitched from overlapping patches." if learned else "High-scoring candidate pixels from the deterministic spectral baseline."),
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
        if learned_answer is not None:
            answer = learned_answer.answer
        elif intent == "REGION_GROUNDING":
            answer = f"The strongest {primary.replace('_', ' ')} candidate region is highlighted. The box covers the spatial extent of pixels passing the evidence threshold."
        elif intent in {"WATER_ANALYSIS", "VEGETATION_ANALYSIS", "BUILT_UP_ANALYSIS"}:
            answer = f"Approximately {stats[f'{primary}_percent']:.1f}% of the scene passes the {primary.replace('_', ' ')} evidence threshold. Review the overlay before treating this as a land-cover map."
        else:
            source = "learned RemoteCLIP evidence" if learned else "the deterministic baseline"
            answer = f"The scene is primarily characterized by {dominant}. {source.capitalize()} estimates {stats['vegetation_percent']:.1f}% vegetation, {stats['built_up_percent']:.1f}% built-up evidence and {stats['water_percent']:.1f}% water evidence."
        confidence = self.confidence.from_probability(probability, threshold, learned=learned)
        models = [learned_landcover_model]
        if learned and intent in {"SINGLE_IMAGE_VQA", "IMAGE_CAPTION", "REGION_GROUNDING"}:
            models.insert(0, "RemoteCLIP RN50 learned vision-language inference")
        elif intent in {"SINGLE_IMAGE_VQA", "IMAGE_CAPTION", "REGION_GROUNDING"}:
            models.insert(0, "Mock Remote-Sensing VLM Adapter")
        capabilities = ["preprocessing", "overlay_generation", "statistics", "confidence"]
        if primary == "water": capabilities.append("water_index")
        elif primary == "vegetation": capabilities.append("vegetation_index")
        elif primary == "built_up": capabilities.append("built_up_index")
        tools = self.tool_registry.names_for(capabilities)
        return answer, stats, evidence, confidence, models, tools, learned

    def _change(self, before: np.ndarray, after: np.ndarray, analysis_id: str, target: str | None):
        after = resize_like(after, before)
        learned = bool(not self.settings.mock_mode and self.changeformer and self.changeformer.available)
        if not learned and not self.settings.mock_mode:
            raise SatQueryError(
                "MODEL_UNAVAILABLE",
                "ChangeFormer V6 and its official source are required for bi-temporal analysis when MOCK_MODE=false.",
                503,
            )
        if learned:
            probability = self.changeformer.predict(before, after)
            self.model_registry.mark_loaded("changeformer")
            change_threshold = 0.5
        else:
            before_float = before.astype(np.float32) / 255.0
            after_float = after.astype(np.float32) / 255.0
            raw = np.mean(np.abs(after_float - before_float), axis=2)
            high = max(float(np.percentile(raw, 98)), 0.08)
            probability = np.clip(raw / high, 0, 1)
            change_threshold = self.settings.change_threshold
        learned_landcover = bool(not self.settings.mock_mode and self.remoteclip and self.remoteclip.available)
        if learned_landcover:
            before_classes, landcover_model = self.remoteclip.landcover_probabilities(before)
            after_classes, _ = self.remoteclip.landcover_probabilities(after)
            self.model_registry.mark_loaded("remoteclip_encoder")
            if self.remoteclip.adapter_available:
                self.model_registry.mark_loaded("satquery_adapter")
            class_thresholds = {label: 0.32 for label in before_classes}
        else:
            before_classes = tiled_dict_predict(before, optical_probabilities, self.settings.tile_size, self.settings.tile_overlap)
            after_classes = tiled_dict_predict(after, optical_probabilities, self.settings.tile_size, self.settings.tile_overlap)
            landcover_model = "Spectral Land-Cover Baseline v1"
            class_thresholds = CLASS_THRESHOLDS
        stats: dict[str, float] = {
            "changed_area_percent": self._coverage(probability, change_threshold),
            "change_threshold": change_threshold,
            "built_up_before_percent": self._coverage(before_classes["built_up"], class_thresholds["built_up"]),
            "built_up_after_percent": self._coverage(after_classes["built_up"], class_thresholds["built_up"]),
            "vegetation_before_percent": self._coverage(before_classes["vegetation"], class_thresholds["vegetation"]),
            "vegetation_after_percent": self._coverage(after_classes["vegetation"], class_thresholds["vegetation"]),
            "water_before_percent": self._coverage(before_classes["water"], class_thresholds["water"]),
            "water_after_percent": self._coverage(after_classes["water"], class_thresholds["water"]),
        }
        stats["built_up_change_pp"] = round(stats["built_up_after_percent"] - stats["built_up_before_percent"], 2)
        stats["vegetation_change_pp"] = round(stats["vegetation_after_percent"] - stats["vegetation_before_percent"], 2)
        stats["water_change_pp"] = round(stats["water_after_percent"] - stats["water_before_percent"], 2)
        output = self.settings.data_dir.resolve() / "outputs" / analysis_id
        overlay_name = "change-overlay.png"
        heatmap(probability, output / "change-heatmap.png")
        overlay_mask(after, probability, output / overlay_name, "change", change_threshold)
        coordinates = bounding_box(probability, change_threshold)
        polygon = largest_polygon(probability, change_threshold)
        evidence_description = (
            "Official ChangeFormer V6 LEVIR change probabilities overlaid on the aligned T2 image."
            if learned
            else "Pixels with material before/after spectral difference overlaid on T2."
        )
        evidence = [
            EvidenceItem(id="change-overlay", type="overlay", label="Change overlay", description=evidence_description, confidence=round(float(probability.mean()), 3), asset_url=self._asset_url(analysis_id, overlay_name), coordinates=coordinates, color="#f55f72"),
            EvidenceItem(id="change-heatmap", type="heatmap", label="Change probability", description=("Learned ChangeFormer probability heatmap." if learned else "Transparent change-evidence heatmap generated from normalized pixel differences."), confidence=round(float(probability.mean()), 3), asset_url=self._asset_url(analysis_id, "change-heatmap.png"), color="#f55f72"),
        ]
        if polygon:
            evidence.append(
                EvidenceItem(
                    id="largest-change-polygon",
                    type="polygon",
                    label="Largest changed region",
                    description="Normalized outline of the largest connected change region.",
                    confidence=round(float(probability[probability >= change_threshold].mean()), 3),
                    asset_url=self._asset_url(analysis_id, overlay_name),
                    coordinates=polygon,
                    color="#f0b45b",
                )
            )
        answer = explain_change(stats, target, learned=learned)
        confidence = self.confidence.from_probability(probability, change_threshold, spatial_quality=0.78, learned=learned)
        tools = self.tool_registry.names_for(["change_detection", "overlay_generation", "statistics", "confidence"])
        change_model = "ChangeFormer V6 LEVIR official-v0.1.0" if learned else "Pixel Change Baseline v1"
        return answer, stats, evidence, confidence, [change_model, landcover_model, "Change Reasoner v1"], tools, learned

    def _cross_modal(self, optical: np.ndarray, sar: np.ndarray, analysis_id: str, target: str | None):
        sar = resize_like(sar, optical)
        learned = bool(not self.settings.mock_mode and self.remoteclip and self.remoteclip.available)
        if learned:
            optical_scores, optical_model = self.remoteclip.landcover_probabilities(optical)
            self.model_registry.mark_loaded("remoteclip_encoder")
            if self.remoteclip.adapter_available:
                self.model_registry.mark_loaded("satquery_adapter")
        else:
            optical_scores = tiled_dict_predict(optical, optical_probabilities, self.settings.tile_size, self.settings.tile_overlap)
            optical_model = "Optical Spectral Baseline v1"
        sar_scores = tiled_dict_predict(sar, sar_probabilities, self.settings.tile_size, self.settings.tile_overlap)
        primary = target if target in {"water", "built_up"} else "water"
        fusion_threshold = 0.32 if learned else CLASS_THRESHOLDS[primary]
        optical_probability = optical_scores[primary]
        sar_probability = sar_scores[primary]
        fusion = self.satfusion.predict(optical_probability, sar_probability, sar_scores["texture"])
        fused = fusion.probability
        agreement = fusion.agreement
        stats = {
            "target_class": primary,
            "optical_evidence_percent": self._coverage(optical_probability, fusion_threshold),
            "sar_evidence_percent": self._coverage(sar_probability, CLASS_THRESHOLDS[primary]),
            "fused_evidence_percent": self._coverage(fused, fusion_threshold),
            "cross_sensor_agreement_percent": round(agreement * 100, 2),
            "optical_mean_probability": round(float(optical_probability.mean()), 3),
            "sar_mean_probability": round(float(sar_probability.mean()), 3),
            "fused_mean_probability": round(float(fused.mean()), 3),
            "optical_fusion_weight": round(fusion.optical_weight, 3),
            "sar_fusion_weight": round(fusion.sar_weight, 3),
        }
        output = self.settings.data_dir.resolve() / "outputs" / analysis_id
        items = []
        for label, image, probability, color in (
            ("optical", optical, optical_probability, "#37b8ff"),
            ("sar", sar, sar_probability, "#f0b45b"),
            ("fused", optical, fused, "#52e0c4"),
        ):
            name = f"{label}-{primary}-evidence.png"
            overlay_mask(image, probability, output / name, primary if label != "fused" else "fused", fusion_threshold)
            items.append(EvidenceItem(id=f"{label}-evidence", type="overlay", label=f"{label.title()} evidence", description=f"{label.title()} contribution to the {primary.replace('_', ' ')} result.", confidence=round(float(probability.mean()), 3), asset_url=self._asset_url(analysis_id, name), color=color))
        answer = (
            f"Optical and SAR evidence were evaluated independently, then fused for {primary.replace('_', ' ')} analysis. "
            f"The fused result marks {stats['fused_evidence_percent']:.1f}% of the scene, with {stats['cross_sensor_agreement_percent']:.1f}% cross-sensor agreement."
        )
        confidence = self.confidence.from_probability(fused, fusion_threshold, agreement=agreement, spatial_quality=0.8, learned=learned)
        tools = self.tool_registry.names_for(["optical_analysis", "sar_analysis", "satfusion", "overlay_generation", "confidence"])
        return answer, stats, items, confidence, [optical_model, "SAR Backscatter + Texture Features v1", self.satfusion.model_name], tools, learned

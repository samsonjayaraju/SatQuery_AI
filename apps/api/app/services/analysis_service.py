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
    LegendItem,
    RegistrationInfo,
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
from app.remote_sensing.alignment import AlignmentResult, align_visual_pair
from app.remote_sensing.change_detection import (
    appearance_change_probability,
    hybrid_change_probability,
    semantic_change_probability,
)
from app.remote_sensing.change_analysis import (
    LAND_COVER_CLASSES,
    land_cover_transitions,
    spatial_change_statistics,
)
from app.remote_sensing.preprocessing import (
    load_visual,
    optical_probabilities,
    resize_like,
    sar_probabilities,
)
from app.remote_sensing.visualization import (
    binary_mask,
    bounding_box,
    draw_box,
    heatmap,
    largest_polygon,
    overlay_mask,
    retain_largest_component,
    save_image,
    transition_overlay,
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

    def _prepare(
        self, paths: list[Path], analysis_id: str
    ) -> tuple[list[np.ndarray], list[str], AlignmentResult | None]:
        output = self.settings.data_dir.resolve() / "outputs" / analysis_id
        output.mkdir(parents=True, exist_ok=True)
        arrays: list[np.ndarray] = []
        urls: list[str] = []
        if len(paths) == 2:
            alignment = align_visual_pair(
                paths[0], paths[1], min_confidence=self.settings.registration_min_confidence
            )
            source_arrays = [alignment.first, alignment.second]
        else:
            source_arrays = [load_visual(paths[0])]
            alignment = None
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
        return arrays, urls, alignment

    @staticmethod
    def _coverage(
        probability: np.ndarray, threshold: float = 0.58, valid_mask: np.ndarray | None = None
    ) -> float:
        selected = probability >= threshold
        if valid_mask is None:
            return round(float(selected.mean() * 100), 2)
        valid = valid_mask.astype(bool)
        if valid.shape != probability.shape:
            valid = np.asarray(
                Image.fromarray(valid.astype(np.uint8) * 255).resize(
                    (probability.shape[1], probability.shape[0]), Image.Resampling.NEAREST
                )
            ) >= 128
        return round(float(np.count_nonzero(selected & valid) / max(int(valid.sum()), 1) * 100), 2)

    @staticmethod
    def _normalize_landcover(probabilities: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        labels = list(probabilities)
        stacked = np.stack([np.clip(probabilities[label], 0, None) for label in labels])
        totals = stacked.sum(axis=0, keepdims=True)
        normalized = np.divide(
            stacked,
            totals,
            out=np.full_like(stacked, 1.0 / max(len(labels), 1)),
            where=totals > 1e-8,
        )
        return {label: normalized[index] for index, label in enumerate(labels)}

    @staticmethod
    def _soft_percentage(probability: np.ndarray) -> float:
        return round(float(np.mean(probability) * 100), 2)

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
        inspection: InspectionResponse = inspect_inputs(paths, mode)
        steps.append(self._step("Input inspection", "tool", started, "Validated raster structure, modality and pair compatibility."))
        if not inspection.valid:
            raise SatQueryError(
                "UNSUPPORTED_COMPOSITE_IMAGE",
                inspection.visual_quality.recommendation or "The selected input is not suitable for raster analysis.",
                422,
            )

        started = time.perf_counter()
        if progress and len(paths) == 2:
            progress("registering", "Registering the image pair and measuring alignment quality")
        arrays, urls, alignment = self._prepare(paths, analysis_id)
        alignment_method = alignment.method if alignment is not None else "not_required"
        for index, (metadata, array) in enumerate(zip(inspection.images, arrays)):
            metadata.thumbnail_url = urls[index]
            metadata.display_width = array.shape[1]
            metadata.display_height = array.shape[0]
        if alignment is not None:
            registration_status = (
                "accepted" if alignment.confidence >= self.settings.registration_min_confidence else "low_quality"
            )
            registration_warnings = list(alignment.warnings)
            if registration_status == "low_quality":
                registration_warnings.append(
                    "Input alignment quality is insufficient for highly confident quantitative change estimation."
                )
            inspection.registration = RegistrationInfo(
                method=alignment.method,
                confidence=alignment.confidence,
                status=registration_status,
                transform=alignment.transform,
                warnings=registration_warnings,
            )
            for warning in registration_warnings:
                if warning not in inspection.warnings:
                    inspection.warnings.append(warning)
            steps.append(
                self._step(
                    "Image registration",
                    "tool",
                    started,
                    f"Applied {alignment.method} with {alignment.confidence:.0%} registration confidence.",
                )
            )
        else:
            steps.append(self._step("Image preparation", "tool", started, "Prepared the single raster without pair registration."))

        started = time.perf_counter()
        intent = self.interpreter.classify(query, mode)
        steps.append(self._step("Query interpretation", "system", started, f"Detected {intent.intent} at {intent.confidence:.0%} routing confidence."))

        started = time.perf_counter()
        if progress:
            if not self.settings.mock_mode:
                progress("loading_model", "Loading the selected local specialist model")
            progress("processing", f"Executing specialist workflow for {intent.intent}")
        transitions = []
        if mode == "bi_temporal":
            answer, stats, evidence, confidence, models, tools, learned, transitions = self._change(
                arrays[0],
                arrays[1],
                analysis_id,
                intent.entities.get("target_class"),
                inspection.visual_quality.score,
                alignment_method,
                alignment.valid_mask if alignment is not None else None,
                inspection,
                inspection.registration.confidence,
            )
        elif mode == "cross_modal":
            answer, stats, evidence, confidence, models, tools, learned = self._cross_modal(
                arrays[0], arrays[1], analysis_id, intent.entities.get("target_class"),
                inspection.visual_quality.score, inspection.registration.confidence,
            )
        else:
            answer, stats, evidence, confidence, models, tools, learned = self._single(
                arrays[0], analysis_id, query, intent.intent, intent.entities.get("target_class"), inspection.visual_quality.score
            )
        observed_tools = ["InputInspector"]
        if len(paths) == 2:
            observed_tools.extend(["PairCompatibilityChecker", "ImageRegistration"])
        tools = list(dict.fromkeys([*observed_tools, *tools]))
        steps.append(self._step("Specialist workflow", "model", started, f"Executed {', '.join(models)} with tiled spatial evidence generation."))

        runtime_ms = max(1, round((time.perf_counter() - total_started) * 1000))
        if progress:
            progress("postprocessing", "Cleaning masks, extracting transitions and spatial statistics")
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
                "registration_confidence": inspection.registration.confidence,
                "registration_threshold": self.settings.registration_min_confidence,
                "change_threshold": stats.get("change_threshold", self.settings.change_threshold),
                "device": self.model_registry.device,
            },
            steps=steps,
            runtime_ms=runtime_ms,
            status="success",
            mock_mode=self.settings.mock_mode,
            warnings=inspection.warnings,
        )
        result = AnalysisResponse(
            analysis_id=analysis_id,
            created_at=datetime.now(timezone.utc),
            task=intent.intent,
            query=query,
            answer=answer,
            development_label=(
                "DEVELOPMENT MOCK OUTPUT · deterministic local baseline; learned checkpoints were not invoked."
                if not learned
                else None
            ),
            confidence=confidence,
            evidence=evidence,
            statistics=stats,
            transitions=transitions,
            warnings=inspection.warnings,
            inspection=inspection,
            execution_trace=trace,
            runtime_ms=runtime_ms,
        )
        self.history.save(result)
        logger.info("analysis completed", extra={"analysis_id": analysis_id, "runtime_ms": runtime_ms})
        return result

    def _single(
        self,
        image: np.ndarray,
        analysis_id: str,
        query: str,
        intent: str,
        target: str | None,
        input_quality: float = 1.0,
    ):
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
        probabilities = self._normalize_landcover(probabilities)
        stats = {f"{label}_percent": self._soft_percentage(value) for label, value in probabilities.items()}
        ranked = sorted(((label, value) for label, value in stats.items()), key=lambda item: float(item[1]), reverse=True)
        learned_answer = None
        if learned and intent in {"SINGLE_IMAGE_VQA", "IMAGE_CAPTION"}:
            learned_answer = self.remoteclip.answer(image, query, target, caption=intent == "IMAGE_CAPTION")
        dominant_label = ranked[0][0].replace("_percent", "")
        caption_label = learned_answer.label if learned_answer and learned_answer.label != "mixed" else None
        primary = target or caption_label or dominant_label
        probability = probabilities.get(primary, probabilities[dominant_label])
        if learned and intent == "REGION_GROUNDING":
            probability, _ = self.remoteclip.ground(image, primary, target)
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
                description=("Learned patch-level RemoteCLIP/adapter evidence; review boundaries before treating it as pixel segmentation." if learned else "High-scoring candidate pixels from the deterministic spectral baseline."),
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
        dominant_share = max(float(stats[ranked[0][0]]), 1e-6)
        semantic_consistency = min(1.0, float(stats.get(f"{primary}_percent", 0.0)) / dominant_share)
        confidence = self.confidence.from_probability(
            probability,
            threshold,
            learned=learned,
            model_score=learned_answer.score if learned_answer else None,
            semantic_consistency=semantic_consistency,
            input_quality=input_quality,
        )
        models = [learned_landcover_model]
        if learned and intent in {"SINGLE_IMAGE_VQA", "IMAGE_CAPTION", "REGION_GROUNDING"}:
            models.insert(0, "RemoteCLIP RN50 learned vision-language inference")
        elif intent in {"SINGLE_IMAGE_VQA", "IMAGE_CAPTION", "REGION_GROUNDING"}:
            models.insert(0, "Mock Remote-Sensing VLM Adapter")
        capabilities = ["preprocessing", "overlay_generation", "statistics", "confidence"]
        tools = self.tool_registry.names_for(capabilities)
        return answer, stats, evidence, confidence, models, tools, learned

    def _change(
        self,
        before: np.ndarray,
        after: np.ndarray,
        analysis_id: str,
        target: str | None,
        input_quality: float = 1.0,
        alignment_method: str = "not_provided",
        valid_mask: np.ndarray | None = None,
        inspection: InspectionResponse | None = None,
        registration_quality: float | None = None,
    ):
        after = resize_like(after, before)
        learned = bool(not self.settings.mock_mode and self.changeformer and self.changeformer.available)
        if not learned and not self.settings.mock_mode:
            raise SatQueryError(
                "MODEL_UNAVAILABLE",
                "ChangeFormer V6 and its official source are required for bi-temporal analysis when MOCK_MODE=false.",
                503,
            )
        if learned:
            structural_probability = self.changeformer.predict(before, after)
            self.model_registry.mark_loaded("changeformer")
        else:
            before_float = before.astype(np.float32) / 255.0
            after_float = after.astype(np.float32) / 255.0
            raw = np.mean(np.abs(after_float - before_float), axis=2)
            high = max(float(np.percentile(raw, 98)), 0.08)
            structural_probability = np.clip(raw / high, 0, 1)
        learned_landcover = bool(not self.settings.mock_mode and self.remoteclip and self.remoteclip.available)
        if learned_landcover:
            before_classes, landcover_model = self.remoteclip.landcover_probabilities(before)
            after_classes, _ = self.remoteclip.landcover_probabilities(after)
            self.model_registry.mark_loaded("remoteclip_encoder")
            if self.remoteclip.adapter_available:
                self.model_registry.mark_loaded("satquery_adapter")
        else:
            before_classes = tiled_dict_predict(before, optical_probabilities, self.settings.tile_size, self.settings.tile_overlap)
            after_classes = tiled_dict_predict(after, optical_probabilities, self.settings.tile_size, self.settings.tile_overlap)
            landcover_model = "Spectral Land-Cover Baseline v1"
        before_classes = self._normalize_landcover(before_classes)
        after_classes = self._normalize_landcover(after_classes)
        appearance_probability = appearance_change_probability(before, after)
        semantic_probability = semantic_change_probability(before_classes, after_classes, target)
        probability, change_method = hybrid_change_probability(
            structural_probability, appearance_probability, semantic_probability, target
        )
        change_threshold = 0.5
        if valid_mask is not None:
            probability = np.where(valid_mask, probability, 0.0).astype(np.float32)
        metadata = inspection.images[0] if inspection and inspection.images else None
        transitions, transition_index, valid = land_cover_transitions(
            before_classes, after_classes, valid_mask, metadata
        )
        stats: dict[str, float | str] = {
            "changed_area_percent": self._coverage(probability, change_threshold, valid),
            "change_threshold": change_threshold,
            "built_up_before_percent": self._soft_percentage(before_classes["built_up"]),
            "built_up_after_percent": self._soft_percentage(after_classes["built_up"]),
            "vegetation_before_percent": self._soft_percentage(before_classes["vegetation"]),
            "vegetation_after_percent": self._soft_percentage(after_classes["vegetation"]),
            "water_before_percent": self._soft_percentage(before_classes["water"]),
            "water_after_percent": self._soft_percentage(after_classes["water"]),
            "bare_land_before_percent": self._soft_percentage(before_classes["bare_land"]),
            "bare_land_after_percent": self._soft_percentage(after_classes["bare_land"]),
            "agriculture_before_percent": self._soft_percentage(before_classes["agriculture"]),
            "agriculture_after_percent": self._soft_percentage(after_classes["agriculture"]),
        }
        for label in ("built_up", "vegetation", "water", "bare_land", "agriculture"):
            stats[f"{label}_change_pp"] = round(
                float(stats[f"{label}_after_percent"]) - float(stats[f"{label}_before_percent"]), 2
            )
        stats["change_method"] = change_method
        stats["structural_change_percent"] = self._coverage(structural_probability, change_threshold, valid)
        stats["appearance_change_percent"] = self._coverage(appearance_probability, change_threshold, valid)
        stats["semantic_change_percent"] = self._coverage(semantic_probability, change_threshold, valid)
        stats["alignment_method"] = alignment_method
        stats["registration_confidence"] = round(float(registration_quality or 0), 3)
        stats.update(spatial_change_statistics(probability, change_threshold, valid, metadata))
        output = self.settings.data_dir.resolve() / "outputs" / analysis_id
        overlay_name = "change-overlay.png"
        heatmap(probability, output / "change-heatmap.png")
        binary_mask(probability, output / "change-mask.png", change_threshold)
        overlay_mask(after, probability, output / overlay_name, "change", change_threshold)
        transition_name = "land-cover-transitions.png"
        transition_overlay(after, transition_index, list(LAND_COVER_CLASSES), output / transition_name)
        coordinates = bounding_box(probability, change_threshold)
        polygon = largest_polygon(probability, change_threshold)
        evidence_description = (
            "ChangeFormer V6 structural probabilities combined with RemoteCLIP/SatQuery land-cover transitions and appearance evidence."
            if learned
            else "Hybrid appearance and land-cover transition evidence overlaid on T2."
        )
        evidence = [
            EvidenceItem(
                id="change-overlay", type="overlay", label="Change overlay",
                description=evidence_description, confidence=round(float(probability[valid].mean()), 3),
                asset_url=self._asset_url(analysis_id, overlay_name), coordinates=coordinates, color="#f55f72",
                legend=[LegendItem(label="Changed", color="#f55f72"), LegendItem(label="Unchanged", color="#00000000")],
            ),
            EvidenceItem(
                id="change-heatmap", type="probability_map", label="Change probability",
                description="Hybrid material-change probability map; opacity increases with model evidence.",
                confidence=round(float(probability[valid].mean()), 3), asset_url=self._asset_url(analysis_id, "change-heatmap.png"), color="#f55f72",
                legend=[LegendItem(label="Low probability", color="#f55f7233"), LegendItem(label="High probability", color="#f55f72")],
            ),
            EvidenceItem(
                id="change-mask", type="mask", label="Binary change mask",
                description=f"Pixels at or above the recorded {change_threshold:.2f} change threshold.",
                confidence=None, asset_url=self._asset_url(analysis_id, "change-mask.png"), color="#f55f72",
                legend=[LegendItem(label="Changed", color="#f55f72"), LegendItem(label="Unchanged", color="#00000000")],
            ),
            EvidenceItem(
                id="transition-overlay", type="transition", label="Semantic transition overlay",
                description="Pixels whose dominant land-cover class changed, coloured by the T2 class.",
                confidence=None, asset_url=self._asset_url(analysis_id, transition_name), color="#52e0c4",
                legend=[
                    LegendItem(label=label.replace("_", " ").title(), color={
                        "water": "#37b8ff", "vegetation": "#5bdc91", "built_up": "#fbb150",
                        "bare_land": "#caa46c", "agriculture": "#bbdd56",
                    }[label]) for label in LAND_COVER_CLASSES
                ],
            ),
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
        answer = explain_change(stats, target, learned=learned, alignment_method=alignment_method)
        confidence = self.confidence.from_probability(
            probability, change_threshold, spatial_quality=0.78, learned=learned,
            input_quality=input_quality, registration_quality=registration_quality,
        )
        tools = self.tool_registry.names_for([
            "registration", "change_detection", "mask_postprocessing", "transition_analysis",
            "overlay_generation", "statistics", "confidence",
        ])
        change_model = (
            "ChangeFormer V6 LEVIR official-v0.1.0 + Environmental Semantic Change v1"
            if learned
            else "Environmental Appearance + Semantic Change v1"
        )
        return answer, stats, evidence, confidence, [change_model, landcover_model, "Change Reasoner v1"], tools, learned, transitions

    def _cross_modal(
        self,
        optical: np.ndarray,
        sar: np.ndarray,
        analysis_id: str,
        target: str | None,
        input_quality: float = 1.0,
        registration_quality: float | None = None,
    ):
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
            "registration_confidence": round(float(registration_quality or 0), 3),
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
        confidence = self.confidence.from_probability(
            fused,
            fusion_threshold,
            agreement=agreement,
            spatial_quality=0.8,
            learned=learned,
            input_quality=input_quality,
            registration_quality=registration_quality,
        )
        tools = self.tool_registry.names_for(["optical_analysis", "sar_analysis", "satfusion", "overlay_generation", "confidence"])
        return answer, stats, items, confidence, [optical_model, "SAR Backscatter + Texture Features v1", self.satfusion.model_name], tools, learned

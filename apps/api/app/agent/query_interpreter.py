from __future__ import annotations

import re

from app.api.schemas.analysis import IntentResult


class QueryInterpreter:
    """Deterministic, inspectable first-pass task classifier."""

    _target_patterns = {
        "water": r"\b(water|river|lake|reservoir|flood|flooded)\b",
        "vegetation": r"\b(vegetation|forest|green|crop|agriculture)\b",
        "built_up": r"\b(built[ -]?up|building|urban|developed|development)\b",
    }

    def classify(self, query: str, input_mode: str) -> IntentResult:
        normalized = " ".join(query.lower().strip().split())
        entities = {
            "target_class": target
            for target, pattern in self._target_patterns.items()
            if re.search(pattern, normalized)
        }
        target = entities.get("target_class")

        if input_mode == "bi_temporal":
            if target and any(word in normalized for word in ("increase", "decrease", "changed", "change")):
                intent = {
                    "water": "WATER_CHANGE",
                    "vegetation": "VEGETATION_CHANGE",
                    "built_up": "BUILT_UP_CHANGE",
                }[target]
                capabilities = ["change_detection", "land_cover", "change_reasoning"]
            elif any(word in normalized for word in ("describe", "what changed", "compare", "change")):
                intent = "CHANGE_DESCRIPTION"
                capabilities = ["change_detection", "land_cover", "change_reasoning"]
            else:
                intent = "BI_TEMPORAL_CHANGE"
                capabilities = ["change_detection", "change_reasoning"]
            return IntentResult(intent=intent, confidence=0.94, entities=entities, required_capabilities=capabilities)

        if input_mode == "cross_modal":
            if target == "water":
                intent, confidence = "OPTICAL_SAR_WATER", 0.97
            elif target == "built_up":
                intent, confidence = "OPTICAL_SAR_BUILT_UP", 0.96
            else:
                intent, confidence = "CROSS_MODAL_ANALYSIS", 0.91
            return IntentResult(
                intent=intent,
                confidence=confidence,
                entities=entities,
                required_capabilities=["optical_analysis", "sar_analysis", "satfusion"],
            )

        if re.search(r"\b(highlight|locate|where|show|outline|mark)\b", normalized):
            return IntentResult(
                intent="REGION_GROUNDING",
                confidence=0.93,
                entities=entities,
                required_capabilities=["grounding", "overlay_generation"],
            )
        if target == "water":
            intent, capabilities = "WATER_ANALYSIS", ["water_index", "land_cover"]
        elif target == "vegetation":
            intent, capabilities = "VEGETATION_ANALYSIS", ["vegetation_index", "land_cover"]
        elif target == "built_up":
            intent, capabilities = "BUILT_UP_ANALYSIS", ["built_up_index", "land_cover"]
        elif re.search(r"\b(describe|caption|summari[sz]e|scene)\b", normalized):
            intent, capabilities = "IMAGE_CAPTION", ["caption", "land_cover"]
        elif re.search(r"\b(land cover|land-cover|classes|percentage)\b", normalized):
            intent, capabilities = "LAND_COVER_ANALYSIS", ["land_cover"]
        elif normalized:
            intent, capabilities = "SINGLE_IMAGE_VQA", ["vqa", "land_cover"]
        else:
            intent, capabilities = "UNKNOWN", []
        return IntentResult(
            intent=intent,
            confidence=0.88 if intent != "UNKNOWN" else 0.2,
            entities=entities,
            required_capabilities=capabilities,
        )

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolDefinition:
    id: str
    name: str
    capabilities: tuple[str, ...]


DEFAULT_TOOLS = (
    ToolDefinition("geotiff_inspector", "GeoTIFFInspector", ("inspection",)),
    ToolDefinition("pair_compatibility", "PairCompatibilityChecker", ("inspection", "registration")),
    ToolDefinition("image_registration", "ImageRegistration", ("registration",)),
    ToolDefinition("image_tiler", "ImageTiler", ("preprocessing",)),
    ToolDefinition("band_selector", "BandSelector", ("preprocessing", "optical_analysis", "land_cover")),
    ToolDefinition("ndvi", "NDVICalculator", ("vegetation_index",)),
    ToolDefinition("ndwi", "NDWICalculator", ("water_index",)),
    ToolDefinition("ndbi", "NDBICalculator", ("built_up_index",)),
    ToolDefinition("sar_analyzer", "SARBackscatterAnalyzer", ("sar_analysis",)),
    ToolDefinition("satfusion", "SatFusion", ("satfusion", "fusion")),
    ToolDefinition("change_mask", "ChangeMaskGenerator", ("change_detection",)),
    ToolDefinition("mask_postprocessor", "MaskPostProcessor", ("mask_postprocessing",)),
    ToolDefinition("landcover_transitions", "LandCoverTransitionAnalyzer", ("transition_analysis",)),
    ToolDefinition("polygon_extractor", "PolygonExtractor", ("overlay_generation",)),
    ToolDefinition("area_calculator", "AreaCalculator", ("statistics",)),
    ToolDefinition("overlay_generator", "OverlayGenerator", ("overlay_generation",)),
    ToolDefinition("confidence", "ConfidenceCalculator", ("confidence",)),
    ToolDefinition("report", "ReportBuilder", ("report",)),
)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools = {tool.id: tool for tool in DEFAULT_TOOLS}

    def select(self, capabilities: list[str]) -> list[ToolDefinition]:
        selected = []
        for tool in self._tools.values():
            if any(capability in tool.capabilities for capability in capabilities):
                selected.append(tool)
        return selected

    def all(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def names_for(self, capabilities: list[str]) -> list[str]:
        return [tool.name for tool in self.select(capabilities)]

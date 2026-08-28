export type InputMode = "single" | "bi_temporal" | "cross_modal";

export const SATQUERY_INTENTS = [
  "SINGLE_IMAGE_VQA",
  "IMAGE_CAPTION",
  "REGION_GROUNDING",
  "LAND_COVER_ANALYSIS",
  "WATER_ANALYSIS",
  "VEGETATION_ANALYSIS",
  "BUILT_UP_ANALYSIS",
  "BI_TEMPORAL_CHANGE",
  "CHANGE_VQA",
  "CHANGE_DESCRIPTION",
  "CROSS_MODAL_ANALYSIS",
  "OPTICAL_SAR_WATER",
  "OPTICAL_SAR_BUILT_UP",
  "UNKNOWN",
] as const;

export type SatQueryIntent = (typeof SATQUERY_INTENTS)[number];

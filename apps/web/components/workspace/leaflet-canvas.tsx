"use client";

import { CRS, type LatLngBoundsExpression } from "leaflet";
import { ImageOverlay, MapContainer } from "react-leaflet";

export default function LeafletCanvas({
  baseUrl,
  evidenceUrl,
  opacity,
  width,
  height,
  resetKey,
}: {
  baseUrl: string;
  evidenceUrl: string | null;
  opacity: number;
  width: number;
  height: number;
  resetKey: number;
}) {
  const normalizedHeight = Math.max(35, (height / Math.max(width, 1)) * 100);
  const bounds: LatLngBoundsExpression = [[0, 0], [normalizedHeight, 100]];
  return (
    <MapContainer
      key={`${baseUrl}-${resetKey}`}
      className="leaflet-canvas"
      crs={CRS.Simple}
      bounds={bounds}
      minZoom={-2}
      maxZoom={5}
      zoomSnap={0.25}
      attributionControl={false}
    >
      <ImageOverlay url={baseUrl} bounds={bounds} opacity={1} />
      {evidenceUrl && <ImageOverlay url={evidenceUrl} bounds={bounds} opacity={opacity} />}
    </MapContainer>
  );
}

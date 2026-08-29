"use client";

import { CRS, type LatLngBoundsExpression } from "leaflet";
import { useEffect, useMemo } from "react";
import { ImageOverlay, MapContainer, useMap } from "react-leaflet";

function ViewportController({ bounds, resetKey }: { bounds: LatLngBoundsExpression; resetKey: number }) {
  const map = useMap();

  useEffect(() => {
    // Fit synchronously without Leaflet's zoom transition. Animated teardown
    // can retain a stale pane element during React/Next fast refreshes.
    map.stop();
    map.invalidateSize({ animate: false });
    map.fitBounds(bounds, { animate: false, padding: [8, 8] });
  }, [bounds, map, resetKey]);

  return null;
}

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
  const bounds = useMemo<LatLngBoundsExpression>(
    () => [[0, 0], [normalizedHeight, 100]],
    [normalizedHeight],
  );
  return (
    <MapContainer
      className="leaflet-canvas"
      crs={CRS.Simple}
      bounds={bounds}
      minZoom={-2}
      maxZoom={5}
      zoomSnap={0.25}
      attributionControl={false}
      zoomAnimation={false}
      fadeAnimation={false}
      markerZoomAnimation={false}
    >
      <ViewportController bounds={bounds} resetKey={resetKey} />
      <ImageOverlay url={baseUrl} bounds={bounds} opacity={1} />
      {evidenceUrl && <ImageOverlay url={evidenceUrl} bounds={bounds} opacity={opacity} />}
    </MapContainer>
  );
}

"""Extract customer-candidate locations from OpenStreetMap buildings."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from shapely.geometry import Point

from ..data.osm_disk_cache import OsmDiskCache, overpass_features_cached

# 1 query covers virtually all cities. Only fall back if the area has
# very few mapped buildings (rare).
_CUSTOMER_TAG_PIPELINE = [
    {"building": True},
    {"shop": True},
]


def _gdf_to_points(gdf) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for geom in gdf.geometry:
        if geom is None:
            continue
        if hasattr(geom, "centroid"):
            c = geom.centroid
        elif isinstance(geom, Point):
            c = geom
        else:
            continue
        lon, lat = float(c.x), float(c.y)
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            pts.append((lat, lon))
    return pts


def extract_building_candidates(
    city: str,
    country: str,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    min_candidates: int = 80,
    disk_cache: Optional[OsmDiskCache] = None,
) -> List[Dict[str, Any]]:
    """
    Extract customer-candidate locations from OSM buildings.

    Uses ``{"building": True}`` which returns all mapped buildings in a
    single Overpass call. Falls back to ``{"shop": True}`` only if the
    first query yields fewer than *min_candidates*.
    """
    seen: set = set()
    out: List[Dict[str, Any]] = []

    for tags in _CUSTOMER_TAG_PIPELINE:
        gdf = overpass_features_cached(
            tags, bbox, city, country, disk_cache, namespace="buildings"
        )
        if gdf is None or len(gdf) == 0:
            continue

        gdf = gdf.reset_index(drop=True)
        for lat, lon in _gdf_to_points(gdf):
            key = (round(lat, 6), round(lon, 6))
            if key in seen:
                continue
            seen.add(key)
            out.append({"id": len(out), "lat": lat, "lon": lon})

        if len(out) >= min_candidates:
            break

    out.sort(key=lambda d: (d["lat"], d["lon"]))
    for new_id, cand in enumerate(out):
        cand["id"] = new_id
    return out

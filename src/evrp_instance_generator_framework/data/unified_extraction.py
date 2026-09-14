"""
Algorithm 1 — Unified Feature Extraction.

One compound Overpass query replaces all separate building, station, and
synthetic-host queries.  The response is partitioned locally (pure in-memory)
into four candidate sets:

    B   — buildings           (customer candidates)
    S1  — observed EV chargers (station priority 1)
    S2  — proxy hosts          (station priority 2)
    S3  — synthetic hosts      (station priority 3)

Complexity: O(1) HTTP call  +  O(F) local scan where F = |features|.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .osm_disk_cache import OsmDiskCache, overpass_features_cached

# All tags combined into one dict → OSMnx generates a single Overpass query.
_COMPOUND_TAGS: Dict[str, Any] = {
    "building": True,
    "amenity": ["charging_station", "fuel", "parking"],
    "shop": "supermarket",
    "landuse": ["commercial", "retail"],
    "highway": "services",
}

# Tag values that map to proxy-host category
_PROXY_AMENITY = {"fuel", "parking"}
_PROXY_LANDUSE = {"commercial", "retail"}

# Tag values that map to synthetic-host category
_SYNTHETIC_AMENITY = {"parking", "service_area"}
_SYNTHETIC_LANDUSE = {"commercial"}
_SYNTHETIC_HIGHWAY = {"services"}


def _parse_first_number(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, str):
        num = ""
        for ch in x.strip():
            if ch.isdigit() or ch == ".":
                num += ch
            else:
                break
        try:
            return float(num) if num else None
        except ValueError:
            return None
    return None


def _parse_green_hint(tags: Dict[str, Any]) -> Optional[int]:
    for key in ("green", "renewable", "environmental_quality", "low_emission"):
        v = tags.get(key)
        if v is None:
            continue
        if isinstance(v, str):
            if v.lower() in {"yes", "true", "1", "green"}:
                return 1
            if v.lower() in {"no", "false", "0"}:
                return 0
    return None


@dataclass
class UnifiedFeatures:
    """Result of the unified compound query, partitioned locally."""
    buildings: List[Dict[str, Any]] = field(default_factory=list)
    ev_stations: List[Dict[str, Any]] = field(default_factory=list)
    proxy_hosts: List[Dict[str, Any]] = field(default_factory=list)
    synthetic_hosts: List[Dict[str, Any]] = field(default_factory=list)


def unified_extract(
    bbox: Optional[Tuple[float, float, float, float]],
    city: str,
    country: str,
    disk_cache: Optional[OsmDiskCache] = None,
) -> UnifiedFeatures:
    """
    Single compound Overpass query → local partitioning into 4 candidate sets.

    This replaces all of:
      - ``customers.extraction.extract_building_candidates``
      - ``stations.extraction.extract_station_candidates``
      - The query part of ``stations.selection.build_synthetic_station_hosts``
    """
    gdf = overpass_features_cached(
        _COMPOUND_TAGS, bbox, city, country, disk_cache,
        namespace="unified",
    )

    result = UnifiedFeatures()
    if gdf is None or len(gdf) == 0:
        return result

    gdf = gdf.reset_index(drop=True)

    seen_b: set = set()
    seen_s1: set = set()
    seen_s2: set = set()
    seen_s3: set = set()

    for _i, row in gdf.iterrows():
        geom = row.get("geometry")
        if geom is None:
            continue
        centroid = geom.centroid if hasattr(geom, "centroid") else geom
        if not (hasattr(centroid, "x") and hasattr(centroid, "y")):
            continue
        lon, lat = float(centroid.x), float(centroid.y)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue

        tags_dict = row.to_dict()
        key = (round(lat, 6), round(lon, 6))

        amenity_val = _str_or_none(tags_dict.get("amenity"))
        building_val = tags_dict.get("building")
        landuse_val = _str_or_none(tags_dict.get("landuse"))
        shop_val = _str_or_none(tags_dict.get("shop"))
        highway_val = _str_or_none(tags_dict.get("highway"))

        # S1: observed EV chargers
        if amenity_val == "charging_station":
            if key not in seen_s1:
                seen_s1.add(key)
                raw_tags = _clean_tags(tags_dict)
                result.ev_stations.append({
                    "id": len(result.ev_stations),
                    "lat": lat, "lon": lon,
                    "is_green_hint": _parse_green_hint(tags_dict),
                    "charging_power_kW_hint": _parse_first_number(
                        tags_dict.get("power") or tags_dict.get("charging_power")
                    ),
                    "num_slots_hint": _safe_int(_parse_first_number(
                        tags_dict.get("capacity") or tags_dict.get("num_slots")
                    )),
                    "station_source_type": "observed_ev",
                    "source_priority": 1,
                    "is_real_observed_ev": True,
                    "osm_tags": raw_tags,
                })

        # S2: proxy hosts (fuel, parking, supermarket, commercial/retail landuse)
        is_proxy = (
            (amenity_val in _PROXY_AMENITY)
            or (landuse_val in _PROXY_LANDUSE)
            or (shop_val == "supermarket")
        )
        if is_proxy and amenity_val != "charging_station":
            if key not in seen_s2:
                seen_s2.add(key)
                raw_tags = _clean_tags(tags_dict)
                result.proxy_hosts.append({
                    "id": len(result.proxy_hosts),
                    "lat": lat, "lon": lon,
                    "is_green_hint": None,
                    "charging_power_kW_hint": None,
                    "num_slots_hint": None,
                    "station_source_type": "proxy_host",
                    "source_priority": 2,
                    "is_real_observed_ev": False,
                    "osm_tags": raw_tags,
                })

        # S3: synthetic hosts (parking, commercial landuse, highway services)
        is_synthetic = (
            (amenity_val in _SYNTHETIC_AMENITY)
            or (landuse_val in _SYNTHETIC_LANDUSE)
            or (highway_val in _SYNTHETIC_HIGHWAY)
        )
        if is_synthetic:
            if key not in seen_s3:
                seen_s3.add(key)
                result.synthetic_hosts.append({
                    "id": len(result.synthetic_hosts),
                    "lat": lat, "lon": lon,
                })

        # B: all buildings
        if building_val is not None and building_val is not False:
            if key not in seen_b:
                seen_b.add(key)
                result.buildings.append({
                    "id": len(result.buildings),
                    "lat": lat, "lon": lon,
                })

    # Deterministic ordering
    result.buildings.sort(key=lambda d: (d["lat"], d["lon"]))
    result.ev_stations.sort(key=lambda d: (d["lat"], d["lon"]))
    result.proxy_hosts.sort(key=lambda d: (d["lat"], d["lon"]))
    result.synthetic_hosts.sort(key=lambda d: (d["lat"], d["lon"]))

    for lst in (result.buildings, result.ev_stations, result.proxy_hosts, result.synthetic_hosts):
        for idx, item in enumerate(lst):
            item["id"] = idx

    return result


def _str_or_none(val: Any) -> Optional[str]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return str(val)


def _safe_int(val: Optional[float]) -> Optional[int]:
    if val is None:
        return None
    return int(val)


def _clean_tags(tags_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v for k, v in tags_dict.items()
        if k != "geometry" and v is not None
        and not (isinstance(v, float) and math.isnan(v))
    }

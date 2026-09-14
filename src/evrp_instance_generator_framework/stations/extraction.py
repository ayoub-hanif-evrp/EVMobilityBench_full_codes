"""OSM station candidate extraction with provenance."""

from typing import Any, Dict, List, Optional, Tuple

from ..data.osm_disk_cache import OsmDiskCache, overpass_features_cached

# 2 queries total: priority-1 (real EV chargers) then priority-2 (proxy hosts).
# All proxy tags are combined into a single Overpass call.
_STATION_TAG_PIPELINE: List[Tuple[Dict[str, Any], str, int]] = [
    ({"amenity": "charging_station"}, "observed_ev", 1),
    ({"amenity": ["fuel", "parking"], "shop": "supermarket"}, "proxy_host", 2),
]


def _parse_first_number(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        num = ""
        for ch in s:
            if ch.isdigit() or ch == ".":
                num += ch
            else:
                break
        if num:
            try:
                return float(num)
            except ValueError:
                return None
    return None


def _parse_green_hint(tags: Dict[str, Any]) -> Optional[int]:
    for key in ["green", "renewable", "environmental_quality", "low_emission"]:
        v = tags.get(key)
        if v is None:
            continue
        if isinstance(v, str):
            if v.lower() in {"yes", "true", "1", "green"}:
                return 1
            if v.lower() in {"no", "false", "0"}:
                return 0
    return None


def extract_station_candidates(
    city: str,
    country: str,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    min_candidates: int = 15,
    disk_cache: Optional[OsmDiskCache] = None,
) -> List[Dict[str, Any]]:
    """
    Progressive extraction of charging-station candidates from OSM.

    Queries priority-1 tags first; if fewer than *min_candidates* are found,
    expands to priority-2 tags.  Stops as soon as the threshold is reached.

    Each returned dict includes provenance fields:
        ``station_source_type``, ``source_priority``,
        ``is_real_observed_ev``, ``osm_tags``.

    Priority-3 synthetic hosts when 1 + 2 are insufficient are built later in
    :func:`evrp_benchmark.stations.selection.select_station_set`.
    """
    import math

    seen: set = set()
    out: List[Dict[str, Any]] = []

    for tags, source_type, priority in _STATION_TAG_PIPELINE:
        gdf = overpass_features_cached(
            tags, bbox, city, country, disk_cache, namespace="stations"
        )
        if gdf is None or len(gdf) == 0:
            continue

        gdf = gdf.reset_index(drop=True)
        for _i, row in gdf.iterrows():
            geom = row.get("geometry")
            if geom is None:
                continue
            centroid = geom.centroid if hasattr(geom, "centroid") else geom
            lon, lat = float(centroid.x), float(centroid.y)

            key = (round(lat, 6), round(lon, 6))
            if key in seen:
                continue
            seen.add(key)

            tags_dict = row.to_dict()

            power_raw = _parse_first_number(
                tags_dict.get("power") or tags_dict.get("charging_power")
            )
            slots_raw = _parse_first_number(
                tags_dict.get("capacity") or tags_dict.get("num_slots")
            )
            power_hint = power_raw if (power_raw is not None and math.isfinite(power_raw)) else None
            slots_hint = int(slots_raw) if (slots_raw is not None and math.isfinite(slots_raw)) else None

            # Capture raw tags for provenance (exclude geometry and internal cols)
            raw_tags = {
                k: v for k, v in tags_dict.items()
                if k != "geometry" and v is not None and not (isinstance(v, float) and math.isnan(v))
            }

            out.append({
                "id": len(out),
                "lat": lat,
                "lon": lon,
                "is_green_hint": _parse_green_hint(tags_dict),
                "charging_power_kW_hint": power_hint,
                "num_slots_hint": slots_hint,
                # Provenance
                "station_source_type": source_type,
                "source_priority": priority,
                "is_real_observed_ev": (source_type == "observed_ev"),
                "osm_tags": raw_tags,
            })

        if len(out) >= min_candidates:
            break

    out.sort(key=lambda d: (d["source_priority"], d["lat"], d["lon"]))
    for new_id, cand in enumerate(out):
        cand["id"] = new_id
    return out

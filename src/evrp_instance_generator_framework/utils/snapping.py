"""
Unified point-to-graph snapping.

Stage in the generation algorithm:
    Input  — raw (lat, lon) points from OSM extraction.
    Output — (node_id, distance_m) for single points, or a list of deduplicated
             dicts for batch snapping.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import osmnx as ox

from ..exceptions import EvrpUserError
from ..types import CustomerCandidate, StationCandidate

_STATION_EXTRA_FIELDS = (
    "is_green_hint",
    "charging_power_kW_hint",
    "num_slots_hint",
    "station_source_type",
    "source_priority",
    "osm_tags",
)


def _ensure_crs(G: nx.DiGraph) -> None:
    if "crs" not in G.graph:
        G.graph["crs"] = "EPSG:4326"


def snap_single_point(
    lat: float,
    lon: float,
    movement_graph: nx.DiGraph,
    max_dist_m: float,
) -> Tuple[int, float]:
    """
    Snap a single WGS-84 point to the nearest graph node.

    Returns ``(node_id, great_circle_distance_m)``.
    Raises ``ValueError`` when the nearest node exceeds *max_dist_m*.
    """
    _ensure_crs(movement_graph)
    node_id = ox.distance.nearest_nodes(movement_graph, X=lon, Y=lat)
    nd = movement_graph.nodes[node_id]
    dist_m = float(ox.distance.great_circle(lat, lon, float(nd["y"]), float(nd["x"])))
    if dist_m > max_dist_m:
        raise ValueError(
            f"Snap distance too large: {dist_m:.1f} m > {max_dist_m:.1f} m. "
            f"Input=({lat:.6f}, {lon:.6f}), nearest node {node_id}="
            f"({nd['y']:.6f}, {nd['x']:.6f})"
        )
    return int(node_id), dist_m


def snap_candidates_batch(
    candidates: List[Dict[str, Any]],
    movement_graph: nx.DiGraph,
    max_dist_m: float,
    extra_fields: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """
    Batch-snap candidate dicts to graph nodes.

    Each input dict must have ``id``, ``lat``, ``lon``.  Additional keys
    listed in *extra_fields* are preserved in the output.  Deduplicates by
    snapped ``movement_node_id``, keeping the candidate with the smallest
    snap distance.

    Returns a sorted list of dicts with at least:
        ``id, lat, lon, movement_node_id, snap_distance_m``
    plus any *extra_fields* copied from the winning candidate.
    """
    if not candidates:
        return []

    _ensure_crs(movement_graph)

    lats = np.asarray([float(c["lat"]) for c in candidates], dtype=np.float64)
    lons = np.asarray([float(c["lon"]) for c in candidates], dtype=np.float64)
    node_ids = np.atleast_1d(
        np.asarray(
            ox.distance.nearest_nodes(movement_graph, X=lons, Y=lats),
            dtype=np.int64,
        )
    )

    by_node: Dict[int, Dict[str, Any]] = {}
    for i, cand in enumerate(candidates):
        lat = float(lats[i])
        lon = float(lons[i])
        snapped_node = int(node_ids[i])
        nd = movement_graph.nodes[snapped_node]
        dist_m = float(ox.distance.great_circle(lat, lon, float(nd["y"]), float(nd["x"])))
        if dist_m > max_dist_m:
            continue

        prev = by_node.get(snapped_node)
        if prev is None or dist_m < prev["snap_distance_m"]:
            rec: Dict[str, Any] = {
                "id": int(cand["id"]),
                "lat": lat,
                "lon": lon,
                "movement_node_id": snapped_node,
                "snap_distance_m": dist_m,
            }
            for k in extra_fields:
                rec[k] = cand.get(k)
            by_node[snapped_node] = rec

    result = list(by_node.values())
    result.sort(key=lambda r: (r["movement_node_id"], r["id"]))
    return result


def snap_customers_to_graph(
    candidates: List[Dict[str, Any]],
    movement_graph: nx.DiGraph,
    max_dist_m: float,
) -> List[CustomerCandidate]:
    """Snap and deduplicate customer candidates, returning typed objects."""
    raw = snap_candidates_batch(candidates, movement_graph, max_dist_m)
    return [
        CustomerCandidate(
            id=r["id"],
            lat=r["lat"],
            lon=r["lon"],
            movement_node_id=r["movement_node_id"],
            snap_distance_m=r["snap_distance_m"],
        )
        for r in raw
    ]


def snap_stations_to_graph(
    candidates: List[Dict[str, Any]],
    movement_graph: nx.DiGraph,
    max_dist_m: float,
) -> List[StationCandidate]:
    """Snap and deduplicate station candidates, returning typed objects."""
    raw = snap_candidates_batch(
        candidates,
        movement_graph,
        max_dist_m,
        extra_fields=_STATION_EXTRA_FIELDS,
    )
    return [
        StationCandidate(
            id=r["id"],
            lat=r["lat"],
            lon=r["lon"],
            movement_node_id=r["movement_node_id"],
            snap_distance_m=r["snap_distance_m"],
            is_green_hint=r.get("is_green_hint"),
            charging_power_kW_hint=r.get("charging_power_kW_hint"),
            num_slots_hint=r.get("num_slots_hint"),
            station_source_type=r.get("station_source_type", "observed_ev"),
            source_priority=r.get("source_priority", 1),
            osm_tags=r.get("osm_tags"),
        )
        for r in raw
    ]


def snap_latlon_to_road(
    lat: float,
    lon: float,
    movement_graph: nx.DiGraph,
    max_dist_m: float,
) -> Tuple[int, float]:
    """
    Snap a WGS84 point to the nearest graph node; returns ``(node_id, great_circle_distance_m)``.

    Raises :class:`EvrpUserError` if the nearest node exceeds *max_dist_m*.
    """
    try:
        return snap_single_point(lat, lon, movement_graph, max_dist_m)
    except ValueError as e:
        raise EvrpUserError(
            "Depot (or point) is too far from the road network for the chosen snap limit.\n"
            f"{e}"
        ) from e

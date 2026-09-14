"""
Algorithm 2 — Batch Preprocessing.

Performs unified feature extraction (Algorithm 1) and batch-snaps all
candidate sets in one pass.  The results are stored on `PipelineState`
so that customer and station selection can consume them without any
additional Overpass queries.

Complexity:
    O(1) HTTP call   (unified extraction)
    O(P log V) snap  (one batched KD-tree lookup for P total points)
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..types import PipelineState
from ..utils.snapping import snap_candidates_batch, snap_customers_to_graph, snap_stations_to_graph
from .unified_extraction import unified_extract

_STATION_EXTRA = (
    "is_green_hint",
    "charging_power_kW_hint",
    "num_slots_hint",
    "station_source_type",
    "source_priority",
    "is_real_observed_ev",
    "osm_tags",
)


def run_unified_extraction_and_snap(state: PipelineState) -> PipelineState:
    """
    One call replaces all separate customer / station / synthetic-host queries.

    After this function, ``state`` carries pre-snapped:
        ``_unified_buildings``    — List[CustomerCandidate]
        ``_unified_ev_stations``  — List[StationCandidate]
        ``_unified_proxy_hosts``  — List[StationCandidate]
        ``_unified_synthetic_hosts`` — List[Dict]  (snapped dicts)
    """
    if state._unified_extracted:
        return state

    config = state.config
    feat = unified_extract(
        bbox=state.bbox,
        city=config.city,
        country=config.country,
        disk_cache=state.disk_cache,
    )

    snap_cust = config.customers_pool_snap_max_dist_m
    snap_stat = config.real_stations_snap_max_dist_m

    state._unified_buildings = snap_customers_to_graph(
        feat.buildings, state.movement_graph, snap_cust,
    )
    state._unified_ev_stations = snap_stations_to_graph(
        feat.ev_stations, state.movement_graph, snap_stat,
    )
    state._unified_proxy_hosts = snap_stations_to_graph(
        feat.proxy_hosts, state.movement_graph, snap_stat,
    )
    state._unified_synthetic_hosts = snap_candidates_batch(
        feat.synthetic_hosts, state.movement_graph, snap_stat,
    )

    state._unified_extracted = True
    return state

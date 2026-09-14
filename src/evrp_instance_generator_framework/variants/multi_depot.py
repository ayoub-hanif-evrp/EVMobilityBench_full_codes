"""
Multi-depot EVRPTW instance generation pipeline.

Algorithm  (paper pseudocode § Multi-Depot EVRPTW Generation)
─────────
    1. Prepare movement graph   — same as classic (shared primitive).
    2. Prepare depot set         — each depot: **facility** (building / user or OSM
                                   centroid) + **snapped** legal drivable node for routing
                                   (vehicle enters/leaves via that node). Additional
                                   facilities from OSM building-centric queries or
                                   manual tuples; farthest-first selection when auto.
    3. Extract customer candidates — OSM buildings inside bbox.
    4. Select customers           — depot-aware: each candidate must be
                                   reachable from at least one depot.
    5. Generate stations          — priority-based selection with provenance.
    6. Assign time windows        — tightness-aware, using *minimum* depot
                                   travel time as the temporal anchor.
    7. Run feasibility            — three-tier report (validity / time / energy);
                                   time uses per-depot forward/return maps; energy
                                   uses direct or one-hop-via-station per depot when enabled.
    8. Assemble instance          — BenchmarkInstance with depot list.

Inputs:  ``GenerationConfig`` (variant="multi_depot_evrptw"), ``EVFeatures``.
Outputs: ``BenchmarkInstance`` with populated ``depots`` list.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..data.countries_data_loader import load_countries_data, resolve_station_defaults
from ..data.batch_preprocess import run_unified_extraction_and_snap
from ..exceptions import EvrpUserError
from ..customers.selection import generate_customers_standard
from ..customers.csv_import import apply_customers_to_state, load_customers_from_csv
from ..feasibility_tests import build_multi_depot_report
from ..road_network.utils import (
    build_disk_cache,
    depot_return_times_to_depot,
    depot_single_source_times,
    download_graph,
    graph_bbox,
    prepare_movement_graph,
    snap_depot,
)
from ..utils.snapping import snap_candidates_batch
from ..stations.selection import select_station_set
from ..service_graph.energy_consumption import compute_energy_matrix
from ..service_graph.pairwise_path_matrices import compute_pairwise_path_matrices
from ..service_graph.service_node_mapping import build_service_nodes
from ..validation.instance_validator import attach_post_finalize_artifacts
from ..types import (
    BenchmarkInstance,
    DepotRecord,
    EVFeatures,
    GenerationConfig,
    InstanceMetadata,
    PipelineState,
    StationRecord,
)


def _country_defaults(config: GenerationConfig) -> Dict[str, Any]:
    bundle = load_countries_data()
    return resolve_station_defaults(bundle, config.country, config.city)


def _great_circle_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import osmnx as ox
    return float(ox.distance.great_circle(lat1, lon1, lat2, lon2))


# ── OSM tags for depot-suitable locations ────────────────────────────────

# Overpass tag passes — building-centric first, then landuse / parking fallbacks.
_DEPOT_TAG_PIPELINE = [
    {"building": ["warehouse", "industrial", "commercial", "retail", "supermarket"]},
    {"landuse": ["commercial", "industrial"]},
    {"amenity": "parking"},
]


def _auto_generate_additional_depot_coords(
    num_depots: int,
    primary_lat: float,
    primary_lon: float,
    bbox: Tuple[float, float, float, float],
    movement_graph,
    snap_max_m: float,
    depot_to_node_time: Dict[int, float],
    disk_cache,
    city: str,
    country: str,
) -> List[Tuple[float, float]]:
    """
    Deterministic depot location generation from OSM commercial / industrial POIs.

    Algorithm:
        1. Query OSM for commercial, industrial, and warehouse POIs.
        2. Snap candidates to graph, filter to reachable from primary depot.
        3. Seed with the candidate at median distance from primary depot.
        4. Iteratively select the candidate farthest from all already-selected
           depots (farthest-first / maximin — deterministic, no randomness).
    """
    from ..data.osm_disk_cache import overpass_features_cached

    min_cand = max(20, num_depots * 5)
    seen: set = set()
    points: List[Dict[str, Any]] = []

    for tags in _DEPOT_TAG_PIPELINE:
        gdf = overpass_features_cached(
            tags, bbox, city, country, disk_cache, namespace="auto_depots",
        )
        if gdf is None or len(gdf) == 0:
            continue
        gdf = gdf.reset_index(drop=True)
        for _i, row in gdf.iterrows():
            geom = row.get("geometry")
            if geom is None:
                continue
            c = geom.centroid if hasattr(geom, "centroid") else geom
            lon, lat = float(c.x), float(c.y)
            key = (round(lat, 6), round(lon, 6))
            if key in seen:
                continue
            seen.add(key)
            points.append({"id": len(points), "lat": lat, "lon": lon})
        if len(points) >= min_cand:
            break

    if len(points) < num_depots:
        raise EvrpUserError(
            f"Not enough depot candidates from OSM: got {len(points)}, "
            f"need {num_depots}. Try a denser city or reduce num_additional_depots."
        )

    snapped = snap_candidates_batch(points, movement_graph, snap_max_m)

    reachable = [s for s in snapped if s["movement_node_id"] in depot_to_node_time]
    if len(reachable) < num_depots:
        reachable = snapped
    if len(reachable) < num_depots:
        raise EvrpUserError(
            f"Only {len(reachable)} depot candidates snapped, need {num_depots}."
        )

    for s in reachable:
        s["_d_primary"] = _great_circle_m(s["lat"], s["lon"], primary_lat, primary_lon)
    reachable.sort(key=lambda s: s["_d_primary"])

    selected: List[Dict[str, Any]] = [reachable[len(reachable) // 2]]
    used_nodes = {selected[0]["movement_node_id"]}

    while len(selected) < num_depots:
        remaining = [s for s in reachable if s["movement_node_id"] not in used_nodes]
        if not remaining:
            break
        all_chosen = [{"lat": primary_lat, "lon": primary_lon}] + selected
        best = max(
            remaining,
            key=lambda s: min(
                _great_circle_m(s["lat"], s["lon"], ch["lat"], ch["lon"])
                for ch in all_chosen
            ),
        )
        selected.append(best)
        used_nodes.add(best["movement_node_id"])

    return [(s["lat"], s["lon"]) for s in selected]


def suggest_additional_depot_facilities(
    city: str,
    country: str,
    movement_graph,
    primary_lat: float,
    primary_lon: float,
    num_additional_depots: int,
    *,
    anchor_period: str = "off_peak",
    snap_max_m: float = 500.0,
    disk_cache=None,
) -> List[Tuple[float, float]]:
    """
    Public helper: same deterministic farthest-first OSM logic as auto multi-depot
    generation, returning suggested ``(lat, lon)`` for each *additional* depot
    (not including the primary). Useful for UIs to pre-fill coordinates.
    """
    if num_additional_depots <= 0:
        return []
    bbox = graph_bbox(movement_graph)
    d0_node, _ = snap_depot(
        float(primary_lat), float(primary_lon), movement_graph, snap_max_m,
    )
    weight_attr = f"{anchor_period}_travel_time_s"
    d0_times = depot_single_source_times(movement_graph, d0_node, weight_attr)
    return _auto_generate_additional_depot_coords(
        num_depots=num_additional_depots,
        primary_lat=float(primary_lat),
        primary_lon=float(primary_lon),
        bbox=bbox,
        movement_graph=movement_graph,
        snap_max_m=float(snap_max_m),
        depot_to_node_time=d0_times,
        disk_cache=disk_cache,
        city=city,
        country=country,
    )


# ── Stage 1–2: graph + depot set ─────────────────────────────────────────

def prepare_graph_and_depots(
    config: GenerationConfig,
    ev_features: EVFeatures,
    movement_graph=None,
) -> PipelineState:
    """
    Prepare graph and snap *all* depots (first config depot + additional).

    Two modes for additional depots:
        **Manual** — ``config.additional_depots`` is non-empty: snap each
        user-provided (lat, lon) to the graph.

        **Auto** — ``config.additional_depots`` is empty: auto-generate
        ``config.num_additional_depots`` positions from OSM commercial /
        industrial POIs using deterministic farthest-first selection.

    Each ``DepotRecord`` stores **facility** coordinates (building / OSM centroid /
    ``GenerationConfig`` pin) plus **snapped** graph coordinates for routing.
    """
    rng = np.random.default_rng(config.seed)
    disk_cache = build_disk_cache(config.osm_cache_enabled, config.osm_cache_dir)

    if movement_graph is None:
        movement_graph = download_graph(
            config.city,
            config.country,
            disk_cache,
            osm_network_type=config.osm_network_type,
            osm_retain_all=config.osm_retain_all,
        )
    movement_graph = prepare_movement_graph(
        movement_graph, elevation_provider=config.node_elevation_provider,
    )
    bbox = graph_bbox(movement_graph)

    snap_max = config.additional_depot_snap_max_dist_m or config.depot_snap_max_dist_m
    open_s = config.additional_depot_time_open_s or config.depot_time_open_s
    close_s = config.additional_depot_time_close_s or config.depot_time_close_s
    weight_attr = f"{config.anchor_period}_travel_time_s"

    # First depot (config.depot_lat / depot_lon — synthesis order index 0)
    d0_node, d0_dist = snap_depot(config.depot_lat, config.depot_lon, movement_graph, config.depot_snap_max_dist_m)
    d0_times = depot_single_source_times(movement_graph, d0_node, weight_attr)
    nd0 = movement_graph.nodes[d0_node]
    d0_lat, d0_lon = float(nd0["y"]), float(nd0["x"])

    depots: List[DepotRecord] = [
        DepotRecord(
            id=0,
            lat=d0_lat,
            lon=d0_lon,
            facility_lat=float(config.depot_lat),
            facility_lon=float(config.depot_lon),
            movement_node_id=d0_node,
            snap_distance_m=d0_dist,
            time_open_s=config.depot_time_open_s,
            time_close_s=config.depot_time_close_s,
            is_primary=False,
        )
    ]

    # Resolve additional depot coordinates (manual or auto)
    additional_coords = list(config.additional_depots)
    if not additional_coords and config.num_additional_depots > 0:
        additional_coords = _auto_generate_additional_depot_coords(
            num_depots=config.num_additional_depots,
            primary_lat=config.depot_lat,
            primary_lon=config.depot_lon,
            bbox=bbox,
            movement_graph=movement_graph,
            snap_max_m=snap_max,
            depot_to_node_time=d0_times,
            disk_cache=disk_cache,
            city=config.city,
            country=config.country,
        )

    # Per-depot single-source times: {depot_node_id: {target_node: time_s}}
    depot_times: Dict[int, Dict[int, float]] = {d0_node: d0_times}

    for idx, (lat, lon) in enumerate(additional_coords, start=1):
        fac_lat, fac_lon = float(lat), float(lon)
        nid, dist = snap_depot(fac_lat, fac_lon, movement_graph, snap_max)
        nd = movement_graph.nodes[nid]
        dep_lat, dep_lon = float(nd["y"]), float(nd["x"])
        depots.append(DepotRecord(
            id=idx,
            lat=dep_lat,
            lon=dep_lon,
            facility_lat=fac_lat,
            facility_lon=fac_lon,
            movement_node_id=nid,
            snap_distance_m=dist,
            time_open_s=open_s,
            time_close_s=close_s,
            is_primary=False,
        ))
        if nid not in depot_times:
            depot_times[nid] = depot_single_source_times(movement_graph, nid, weight_attr)

    # Merge: for each target node, keep *minimum* travel time across depots
    merged_tt: Dict[int, float] = {}
    for dt in depot_times.values():
        for node, t in dt.items():
            if node not in merged_tt or t < merged_tt[node]:
                merged_tt[node] = t

    depot_forward_list = [depot_times[d.movement_node_id] for d in depots]
    depot_return_list = [
        depot_return_times_to_depot(movement_graph, d.movement_node_id, weight_attr)
        for d in depots
    ]

    return PipelineState(
        config=config,
        ev_features=ev_features,
        movement_graph=movement_graph,
        rng=rng,
        disk_cache=disk_cache,
        bbox=bbox,
        depot_node_id=d0_node,
        depot_snap_dist_m=d0_dist,
        depot_to_node_time=merged_tt,
        depots=depots,
        depot_travel_times=depot_forward_list,
        depot_return_times=depot_return_list,
    )


# ── Stage 3–4: depot-aware customer generation ───────────────────────────

def generate_customers(state: PipelineState) -> PipelineState:
    """Extract and select customers (uses merged min-depot travel times)."""
    csv_path = state.config.customer_csv_path
    if csv_path:
        imported = load_customers_from_csv(csv_path)
        return apply_customers_to_state(state, imported)
    return generate_customers_standard(state)


# ── Stage 5: stations ─────────────────────────────────────────────────────

def generate_stations(state: PipelineState) -> PipelineState:
    """
    Select stations from pre-extracted candidates (unified extraction).

    Falls back to legacy per-module extraction if unified data is unavailable.
    """
    config = state.config

    if state._unified_extracted:
        real_candidates = list(state._unified_ev_stations) + list(state._unified_proxy_hosts)
        pre_synth = list(state._unified_synthetic_hosts)
    else:
        from ..stations.extraction import extract_station_candidates
        from ..utils.snapping import snap_stations_to_graph as snap_stations

        default_station = max(30, config.num_stations * 3)
        station_min = (
            max(int(config.station_osm_min_candidates), max(1, config.num_stations))
            if config.station_osm_min_candidates is not None
            else default_station
        )

        raw = extract_station_candidates(
            config.city, config.country,
            bbox=state.bbox, min_candidates=station_min,
            disk_cache=state.disk_cache,
        )
        real_candidates = snap_stations(raw, state.movement_graph, config.real_stations_snap_max_dist_m)
        pre_synth = None

    country_defaults = _country_defaults(config)
    blocked = {int(c.movement_node_id) for c in state.customers}
    blocked.add(int(state.depot_node_id))
    for d in state.depots:
        blocked.add(int(d.movement_node_id))
    state.stations = select_station_set(
        num_stations=config.num_stations,
        real_station_candidates=real_candidates,
        customers=state.customers,
        depot_lat=config.depot_lat,
        depot_lon=config.depot_lon,
        movement_graph=state.movement_graph,
        seed=config.seed,
        config=config,
        country_defaults=country_defaults,
        bbox=state.bbox,
        disk_cache=state.disk_cache,
        pre_snapped_synthetic=pre_synth,
        blocked_node_ids=blocked,
        repair_summary=state.repair_summary,
    )
    return state


# ── Stage 6–8: finalize ──────────────────────────────────────────────────

def finalize(
    state: PipelineState,
    *,
    compute_matrices: bool = True,
    run_energy_feasibility: bool = True,
) -> BenchmarkInstance:
    """Assemble the multi-depot BenchmarkInstance."""
    config = state.config
    ev = state.ev_features
    G = state.movement_graph

    service_nodes = build_service_nodes(
        depot_node_id=state.depot_node_id,
        customers=state.customers, stations=state.stations,
    )

    distance_matrix_m = None
    travel_time_matrices_s: Dict[Any, Any] = {}
    energy_matrix_kwh = None
    feasibility: Dict[str, Any] = {}
    feas_level = "validity_time_energy"

    if compute_matrices:
        periods = (config.energy_period,)
        travel_time_matrices_s, distance_matrix_m = compute_pairwise_path_matrices(
            G, service_nodes, periods=periods,
        )
        for p in periods:
            travel_time_matrices_s[p] = travel_time_matrices_s[p] / ev.speed_multiplier

        energy_matrix_kwh = compute_energy_matrix(
            G, service_nodes, period=config.energy_period,
            ev_features=ev,
            precomputed_travel_time=travel_time_matrices_s.get(config.energy_period),
        )

        feasibility = build_multi_depot_report(
            movement_graph=G,
            config=config,
            ev_features=ev,
            depot_node_id=state.depot_node_id,
            depots=state.depots,
            depot_forward=state.depot_travel_times,
            depot_return=state.depot_return_times,
            customers=state.customers,
            stations=state.stations,
            travel_time_matrix_s=travel_time_matrices_s[config.energy_period],
            energy_matrix_kwh=energy_matrix_kwh,
            compute_matrices=True,
            run_energy_feasibility=True,
            service_nodes=service_nodes,
        )
    elif run_energy_feasibility:
        periods = (config.energy_period,)
        tt_tmp, _ = compute_pairwise_path_matrices(
            G, service_nodes, periods=periods, include_distance_matrix=False,
        )
        for p in periods:
            tt_tmp[p] = tt_tmp[p] / ev.speed_multiplier
        energy_tmp = compute_energy_matrix(
            G, service_nodes, period=config.energy_period,
            ev_features=ev, precomputed_travel_time=tt_tmp.get(config.energy_period),
        )
        feasibility = build_multi_depot_report(
            movement_graph=G,
            config=config,
            ev_features=ev,
            depot_node_id=state.depot_node_id,
            depots=state.depots,
            depot_forward=state.depot_travel_times,
            depot_return=state.depot_return_times,
            customers=state.customers,
            stations=state.stations,
            travel_time_matrix_s=tt_tmp[config.energy_period],
            energy_matrix_kwh=energy_tmp,
            compute_matrices=False,
            run_energy_feasibility=True,
            service_nodes=service_nodes,
        )
    else:
        feasibility = build_multi_depot_report(
            movement_graph=G,
            config=config,
            ev_features=ev,
            depot_node_id=state.depot_node_id,
            depots=state.depots,
            depot_forward=state.depot_travel_times,
            depot_return=state.depot_return_times,
            customers=state.customers,
            stations=state.stations,
            travel_time_matrix_s=None,
            energy_matrix_kwh=None,
            compute_matrices=False,
            run_energy_feasibility=False,
            service_nodes=service_nodes,
        )

    n_obs = sum(1 for s in state.stations if s.station_source_type == "observed_ev")
    n_proxy = sum(1 for s in state.stations if s.station_source_type == "proxy_host")
    n_synth = sum(1 for s in state.stations if s.station_source_type == "synthetic")

    metadata = InstanceMetadata(
        city=config.city, country=config.country, seed=config.seed,
        movement_node_count=int(G.number_of_nodes()),
        service_node_count=len(service_nodes),
        variant="multi_depot_evrptw",
        time_window_tightness=config.time_window_tightness,
        feasibility_level=feas_level,
        depot_count=len(state.depots),
        customer_count=len(state.customers),
        station_count_observed_ev=n_obs,
        station_count_proxy_host=n_proxy,
        station_count_synthetic=n_synth,
        elevation_enabled=(config.node_elevation_provider != "none"),
        extra={
            "depot_facility_lat": float(config.depot_lat),
            "depot_facility_lon": float(config.depot_lon),
            "depot_snap_distance_m": float(state.depot_snap_dist_m),
        },
    )

    instance = BenchmarkInstance(
        metadata=metadata, config=config,
        movement_graph=G, service_nodes=service_nodes,
        customers=state.customers, stations=state.stations,
        depot_node_id=state.depot_node_id,
        distance_matrix_m=distance_matrix_m,
        travel_time_matrices_s=travel_time_matrices_s,
        energy_matrix_kwh=energy_matrix_kwh,
        feasibility=feasibility,
        depots=state.depots,
    )
    return attach_post_finalize_artifacts(instance, state.repair_summary)


# ── One-shot entry point ─────────────────────────────────────────────────

def generate_multi_depot_evrptw(
    config: GenerationConfig,
    ev_features: Optional[EVFeatures] = None,
    movement_graph=None,
    compute_matrices: bool = True,
    run_energy_feasibility: bool = True,
) -> BenchmarkInstance:
    """Full multi-depot EVRPTW generation pipeline in one call."""
    if ev_features is None:
        ev_features = EVFeatures()

    state = prepare_graph_and_depots(config, ev_features, movement_graph)
    state = run_unified_extraction_and_snap(state)
    state = generate_customers(state)
    state = generate_stations(state)
    return finalize(state, compute_matrices=compute_matrices, run_energy_feasibility=run_energy_feasibility)

"""
Classic EVRPTW instance generation pipeline.

Algorithm  (paper pseudocode § Classic EVRPTW Generation)
─────────
    1. Prepare movement graph   — download, SCC, elevation, edge attributes.
    2. Prepare depot             — ``config.depot_lat``/``depot_lon`` = **facility**
                                   (building / site); snap to nearest **legal drivable**
                                   node for routing (vehicle access via that node).
    3. Extract customer candidates — OSM buildings inside bbox.
    4. Select customers           — clustered / random / mixed, with spatial distribution.
    5. Generate stations          — priority-based selection with provenance.
    6. Assign time windows        — tightness-aware (wide / medium / tight).
    7. Run feasibility            — validity, time windows, energy (three-tier report).
    8. Assemble instance          — BenchmarkInstance with full metadata.

Inputs:  ``GenerationConfig`` (variant="classic_evrptw"), ``EVFeatures``.
Outputs: ``BenchmarkInstance``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..data.countries_data_loader import load_countries_data, resolve_station_defaults
from ..data.batch_preprocess import run_unified_extraction_and_snap
from ..customers.selection import generate_customers_standard
from ..customers.csv_import import apply_customers_to_state, load_customers_from_csv
from ..feasibility_tests import build_classic_report
from ..road_network.utils import (
    build_disk_cache,
    download_graph,
    graph_bbox,
    prepare_movement_graph,
    snap_depot,
    depot_single_source_times,
)
from ..stations.selection import select_station_set
from ..service_graph.energy_consumption import compute_energy_matrix
from ..service_graph.pairwise_path_matrices import compute_pairwise_path_matrices
from ..service_graph.service_node_mapping import build_service_nodes
from ..validation.instance_validator import attach_post_finalize_artifacts
from ..types import (
    BenchmarkInstance,
    EVFeatures,
    GenerationConfig,
    InstanceMetadata,
    PipelineState,
    StationRecord,
)


# ── Helper: country defaults ──────────────────────────────────────────────

def _country_defaults(config: GenerationConfig) -> Dict[str, Any]:
    bundle = load_countries_data()
    return resolve_station_defaults(bundle, config.country, config.city)


# ── Stage 1–2: graph + depot ─────────────────────────────────────────────

def prepare_graph_and_depot(
    config: GenerationConfig,
    ev_features: EVFeatures,
    movement_graph=None,
) -> PipelineState:
    """
    Download / prepare the movement graph, snap the **depot facility** to the nearest
    drivable vertex, and compute depot→all single-source travel times from that node.
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

    depot_node_id, depot_snap_dist_m = snap_depot(
        config.depot_lat, config.depot_lon,
        movement_graph, config.depot_snap_max_dist_m,
    )

    weight_attr = f"{config.anchor_period}_travel_time_s"
    depot_to_node_time = depot_single_source_times(
        movement_graph, depot_node_id, weight_attr,
    )

    return PipelineState(
        config=config,
        ev_features=ev_features,
        movement_graph=movement_graph,
        rng=rng,
        disk_cache=disk_cache,
        bbox=bbox,
        depot_node_id=depot_node_id,
        depot_snap_dist_m=depot_snap_dist_m,
        depot_to_node_time=depot_to_node_time,
    )


# ── Stage 3–4: customer extraction + selection ───────────────────────────

def generate_customers(state: PipelineState) -> PipelineState:
    """Extract OSM building candidates, snap, and select customers (c/r/rc)."""
    csv_path = state.config.customer_csv_path
    if csv_path:
        imported = load_customers_from_csv(csv_path)
        return apply_customers_to_state(state, imported)
    return generate_customers_standard(state)


# ── Stage 5: station selection with provenance ────────────────────────────

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

        station_min = max(30, config.num_stations * 3)
        if config.station_osm_min_candidates is not None:
            station_min = max(int(config.station_osm_min_candidates), max(1, config.num_stations))

        raw = extract_station_candidates(
            config.city, config.country,
            bbox=state.bbox, min_candidates=station_min,
            disk_cache=state.disk_cache,
        )
        real_candidates = snap_stations(raw, state.movement_graph, config.real_stations_snap_max_dist_m)
        pre_synth = None

    country_defaults = _country_defaults(config)
    blocked = {int(state.depot_node_id)} | {int(c.movement_node_id) for c in state.customers}
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


# ── Stage 6–8: matrices, feasibility, assembly ───────────────────────────

def finalize(
    state: PipelineState,
    *,
    compute_matrices: bool = True,
    run_energy_feasibility: bool = True,
) -> BenchmarkInstance:
    """
    Build service nodes, optionally compute matrices, run three-tier feasibility
    (validity / time / energy), and assemble the final ``BenchmarkInstance``.
    """
    config = state.config
    ev = state.ev_features
    G = state.movement_graph

    service_nodes = build_service_nodes(
        depot_node_id=state.depot_node_id,
        customers=state.customers,
        stations=state.stations,
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

        feasibility = build_classic_report(
            movement_graph=G,
            config=config,
            ev_features=ev,
            depot_node_id=state.depot_node_id,
            customers=state.customers,
            stations=state.stations,
            travel_time_matrix_s=travel_time_matrices_s[config.energy_period],
            energy_matrix_kwh=energy_matrix_kwh,
            depot_to_node_time=state.depot_to_node_time,
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

        feasibility = build_classic_report(
            movement_graph=G,
            config=config,
            ev_features=ev,
            depot_node_id=state.depot_node_id,
            customers=state.customers,
            stations=state.stations,
            travel_time_matrix_s=tt_tmp[config.energy_period],
            energy_matrix_kwh=energy_tmp,
            depot_to_node_time=state.depot_to_node_time,
            compute_matrices=False,
            run_energy_feasibility=True,
            service_nodes=service_nodes,
        )

    else:
        feasibility = build_classic_report(
            movement_graph=G,
            config=config,
            ev_features=ev,
            depot_node_id=state.depot_node_id,
            customers=state.customers,
            stations=state.stations,
            travel_time_matrix_s=None,
            energy_matrix_kwh=None,
            depot_to_node_time=state.depot_to_node_time,
            compute_matrices=False,
            run_energy_feasibility=False,
            service_nodes=service_nodes,
        )

    # Station provenance counts
    n_obs = sum(1 for s in state.stations if s.station_source_type == "observed_ev")
    n_proxy = sum(1 for s in state.stations if s.station_source_type == "proxy_host")
    n_synth = sum(1 for s in state.stations if s.station_source_type == "synthetic")

    metadata = InstanceMetadata(
        city=config.city,
        country=config.country,
        seed=config.seed,
        movement_node_count=int(G.number_of_nodes()),
        service_node_count=len(service_nodes),
        variant="classic_evrptw",
        time_window_tightness=config.time_window_tightness,
        feasibility_level=feas_level,
        depot_count=1,
        satellite_count=0,
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
        metadata=metadata,
        config=config,
        movement_graph=G,
        service_nodes=service_nodes,
        customers=state.customers,
        stations=state.stations,
        depot_node_id=state.depot_node_id,
        distance_matrix_m=distance_matrix_m,
        travel_time_matrices_s=travel_time_matrices_s,
        energy_matrix_kwh=energy_matrix_kwh,
        feasibility=feasibility,
    )
    return attach_post_finalize_artifacts(instance, state.repair_summary)


# ── One-shot entry point ─────────────────────────────────────────────────

def generate_classic_evrptw(
    config: GenerationConfig,
    ev_features: Optional[EVFeatures] = None,
    movement_graph=None,
    compute_matrices: bool = True,
    run_energy_feasibility: bool = True,
) -> BenchmarkInstance:
    """
    Full classic EVRPTW generation pipeline in one call.

    Equivalent to:
        state = prepare_graph_and_depot(config, ev_features, movement_graph)
        state = generate_customers(state)
        state = generate_stations(state)
        instance = finalize(state, compute_matrices=..., run_energy_feasibility=...)
    """
    if ev_features is None:
        ev_features = EVFeatures()

    state = prepare_graph_and_depot(config, ev_features, movement_graph)
    state = run_unified_extraction_and_snap(state)
    state = generate_customers(state)
    state = generate_stations(state)
    return finalize(state, compute_matrices=compute_matrices, run_energy_feasibility=run_energy_feasibility)

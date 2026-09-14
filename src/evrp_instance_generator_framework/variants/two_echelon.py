"""
Two-echelon EVRP instance generation pipeline.

Algorithm  (same backbone as classic / multi-depot: one OSM graph, facility snap,
unified extraction, then variant-specific facilities and coupling.)

    1. Prepare movement graph — download/cache, SCC, elevation, edge speeds (as classic).
    2. Prepare depot — ``config.depot_lat``/``lon`` facility → nearest drivable node;
       depot→all Dijkstra times (as classic).
    3. Unified OSM extraction — shared batch preprocess for buildings / stations
       (as classic).
    4. Setup satellites — manual snap of ``satellite_locations``, **or** OSM logistics
       POIs + deterministic **stratified** selection (distance band around the depot +
       even angular spread) so hubs sit in logical suburban / industrial rings instead
       of being pushed to bbox corners by maximin dispersion.
    5. Generate customers — ``generate_customers_standard`` (identical to classic;
       includes tightness-aware time windows).
    6. Assign customers → satellites — each customer to the **nearest** hub
       (great-circle; ties by lower satellite id). **Capacity lift:** each satellite’s
       ``capacity`` is set to ``max`` (prior floor from config / heuristic, **realized**
       assigned demand), so second-echelon nominal flow is feasible by construction.
    7. Generate stations — ``select_station_set`` (identical to classic).
    8. Finalize — service-node ordering, matrices, three-tier feasibility + satellite reachability,
       ``BenchmarkInstance`` with ``satellites``.

Inputs:  ``GenerationConfig`` (variant="two_echelon_evrp"), ``EVFeatures``.
Outputs: ``BenchmarkInstance`` with populated ``satellites`` list.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..data.countries_data_loader import load_countries_data, resolve_station_defaults
from ..data.batch_preprocess import run_unified_extraction_and_snap
from ..exceptions import EvrpUserError
from ..customers.selection import generate_customers_standard
from ..customers.csv_import import (
    apply_customers_to_state,
    load_customers_from_csv,
    resolve_num_customers_from_config,
)
from ..feasibility_tests import build_two_echelon_report
from ..road_network.utils import (
    build_disk_cache,
    depot_single_source_times,
    download_graph,
    graph_bbox,
    prepare_movement_graph,
    snap_depot,
)
from ..utils.snapping import (
    snap_candidates_batch,
    snap_stations_to_graph as snap_stations,
)
from ..stations.extraction import extract_station_candidates
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
    SatelliteRecord,
)


def _country_defaults(config: GenerationConfig) -> Dict[str, Any]:
    bundle = load_countries_data()
    return resolve_station_defaults(bundle, config.country, config.city)


def _great_circle_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import osmnx as ox
    return float(ox.distance.great_circle(lat1, lon1, lat2, lon2))


def _bearing_rad_from_anchor(
    anchor_lat: float, anchor_lon: float, lat: float, lon: float,
) -> float:
    """
    Local tangent-plane bearing (radians, east-of-north style via atan2(E, N)).

    Adequate for intra-city satellite spacing; used only to spread hubs around the depot.
    """
    dlat = math.radians(lat - anchor_lat)
    dlon = math.radians(lon - anchor_lon)
    north = dlat
    east = dlon * math.cos(math.radians(anchor_lat))
    return math.atan2(east, north)


# ── Stage 1–2: graph + depot ─────────────────────────────────────────────

def prepare_graph_and_depot(
    config: GenerationConfig,
    ev_features: EVFeatures,
    movement_graph=None,
) -> PipelineState:
    """Download / prepare graph and snap the depot **facility** to its road access node."""
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

    d_node, d_dist = snap_depot(
        config.depot_lat, config.depot_lon,
        movement_graph, config.depot_snap_max_dist_m,
    )
    weight_attr = f"{config.anchor_period}_travel_time_s"
    d_times = depot_single_source_times(movement_graph, d_node, weight_attr)

    return PipelineState(
        config=config, ev_features=ev_features,
        movement_graph=movement_graph, rng=rng, disk_cache=disk_cache,
        bbox=bbox, depot_node_id=d_node, depot_snap_dist_m=d_dist,
        depot_to_node_time=d_times,
    )


# ── Stage 3: setup satellites (manual or auto-generated) ─────────────────

# Single Overpass query for all satellite-suitable features
_SATELLITE_TAG_PIPELINE = [
    {
        "amenity": ["parking", "bus_station"],
        "landuse": ["commercial", "industrial"],
        "building": "warehouse",
    },
]


def _select_satellite_site_dicts(
    movement_graph: Any,
    *,
    city: str,
    country: str,
    bbox: Tuple[float, float, float, float],
    disk_cache: Any,
    num_satellites: int,
    primary_lat: float,
    primary_lon: float,
    depot_to_node_time: Dict[int, float],
    satellite_snap_max_dist_m: float,
) -> List[Dict[str, Any]]:
    """
    OSM candidate harvest + snap + stratified selection (depot distance band + bearings).

    Shared by :func:`_auto_generate_satellites` and :func:`suggest_satellite_facility_latlons`.
    """
    if num_satellites <= 0:
        raise EvrpUserError("num_satellites must be positive for satellite selection.")

    from ..data.osm_disk_cache import overpass_features_cached

    min_cand = max(30, num_satellites * 5)
    seen: set = set()
    sat_points: List[Dict[str, Any]] = []

    for tags in _SATELLITE_TAG_PIPELINE:
        gdf = overpass_features_cached(
            tags, bbox, city, country,
            disk_cache, namespace="satellites",
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
            sat_points.append({"id": len(sat_points), "lat": lat, "lon": lon})
        if len(sat_points) >= min_cand:
            break

    if len(sat_points) < num_satellites:
        raise EvrpUserError(
            f"Not enough satellite candidates from OSM: got {len(sat_points)}, "
            f"need {num_satellites}. Try a denser city or reduce num_satellites."
        )

    snapped = snap_candidates_batch(
        sat_points, movement_graph, satellite_snap_max_dist_m,
    )

    reachable = [s for s in snapped if s["movement_node_id"] in depot_to_node_time]
    if len(reachable) < num_satellites:
        reachable = snapped
    if len(reachable) < num_satellites:
        raise EvrpUserError(
            f"Only {len(reachable)} satellite candidates snapped within "
            f"{satellite_snap_max_dist_m} m, need {num_satellites}."
        )

    selected = _stratified_satellite_select(
        reachable,
        anchor_lat=primary_lat,
        anchor_lon=primary_lon,
        num_select=num_satellites,
    )
    if len(selected) < num_satellites:
        raise EvrpUserError(
            f"Could not pick {num_satellites} distinct satellite road nodes "
            f"(got {len(selected)}). Reduce num_satellites or widen the area."
        )
    return selected


def suggest_satellite_facility_latlons(
    movement_graph: Any,
    *,
    city: str,
    country: str,
    bbox: Tuple[float, float, float, float],
    disk_cache: Any,
    num_satellites: int,
    primary_lat: float,
    primary_lon: float,
    depot_to_node_time: Dict[int, float],
    satellite_snap_max_dist_m: float = 200.0,
) -> List[Tuple[float, float]]:
    """
    Propose ``num_satellites`` satellite **facility** (lat, lon) sites for UI / manual mode.

    Uses the same OSM + stratified hub logic as automatic instance generation. Intended
    for web demos that suggest satellite pins around the primary depot.
    """
    sel = _select_satellite_site_dicts(
        movement_graph,
        city=city,
        country=country,
        bbox=bbox,
        disk_cache=disk_cache,
        num_satellites=num_satellites,
        primary_lat=primary_lat,
        primary_lon=primary_lon,
        depot_to_node_time=depot_to_node_time,
        satellite_snap_max_dist_m=satellite_snap_max_dist_m,
    )
    return [(float(s["lat"]), float(s["lon"])) for s in sel]


def setup_satellites(state: PipelineState) -> PipelineState:
    """
    Setup satellite transfer facilities.

    Two modes:
        **Manual** — ``config.satellite_locations`` is non-empty: snap each
        user-provided (lat, lon) to the graph.

        **Auto** — ``config.satellite_locations`` is empty: query OSM for
        logistics / commercial / parking POIs, snap to graph, then select
        ``config.num_satellites`` using **stratified** placement: keep sites in a
        mid-distance band from the depot (not hugging the depot, not the farthest
        outliers), then pick hubs at evenly spaced **bearings** so they ring the city
        core instead of clustering on one side or at bbox corners.

    Auto-generation algorithm:
        1. Query OSM for parking, bus stations, commercial/industrial zones,
           and warehouses within the graph bbox.
        2. Snap all candidate POIs to the road graph.
        3. Filter to candidates reachable from the primary depot (fallback: all snapped).
        4. ``_stratified_satellite_select``: quantile band on depot distance + greedy
           match to evenly spaced target bearings; optional ``_farthest_first_select``
           backfill if the band is too sparse.
        5. Initial ``capacity`` floor = ``ceil(expected_demand / |S| * 1.3)`` (raised later
           per hub when customers are assigned).
    """
    config = state.config

    if config.satellite_locations:
        return _snap_manual_satellites(state)
    return _auto_generate_satellites(state)


def _snap_manual_satellites(state: PipelineState) -> PipelineState:
    """Snap user-provided satellite coordinates to graph nodes."""
    config = state.config
    from ..utils.snapping import snap_single_point

    num_sat = len(config.satellite_locations)
    base_cap = _satellite_capacity(config, num_sat)

    satellites: List[SatelliteRecord] = []
    for idx, (lat, lon) in enumerate(config.satellite_locations):
        try:
            node_id, dist_m = snap_single_point(
                lat, lon, state.movement_graph, config.satellite_snap_max_dist_m,
            )
        except ValueError:
            raise EvrpUserError(
                f"Satellite {idx} at ({lat:.6f}, {lon:.6f}) is too far from the "
                f"road network (max {config.satellite_snap_max_dist_m} m)."
            )
        satellites.append(SatelliteRecord(
            id=idx, lat=lat, lon=lon,
            movement_node_id=node_id, snap_distance_m=dist_m,
            capacity=base_cap,
            time_open_s=config.depot_time_open_s,
            time_close_s=config.depot_time_close_s,
        ))

    state.satellites = satellites
    return state


def _auto_generate_satellites(state: PipelineState) -> PipelineState:
    """
    Deterministic satellite placement from OSM POIs.

    Uses stratified OSM hub selection (distance band + bearings around the depot).
    """
    config = state.config
    num_sat = config.num_satellites

    if num_sat <= 0:
        raise EvrpUserError(
            "Two-echelon variant requires num_satellites > 0 "
            "(or provide satellite_locations for manual mode)."
        )

    selected = _select_satellite_site_dicts(
        state.movement_graph,
        city=config.city,
        country=config.country,
        bbox=state.bbox,
        disk_cache=state.disk_cache,
        num_satellites=num_sat,
        primary_lat=config.depot_lat,
        primary_lon=config.depot_lon,
        depot_to_node_time=state.depot_to_node_time,
        satellite_snap_max_dist_m=float(config.satellite_snap_max_dist_m),
    )

    base_cap = _satellite_capacity(config, num_sat)
    satellites: List[SatelliteRecord] = []
    for idx, s in enumerate(selected):
        satellites.append(SatelliteRecord(
            id=idx, lat=s["lat"], lon=s["lon"],
            movement_node_id=s["movement_node_id"],
            snap_distance_m=s["snap_distance_m"],
            capacity=base_cap,
            time_open_s=config.depot_time_open_s,
            time_close_s=config.depot_time_close_s,
        ))

    state.satellites = satellites
    return state


def _satellite_capacity(config: GenerationConfig, num_sat: int) -> int:
    if config.satellite_capacity is not None:
        return int(config.satellite_capacity)
    n_customers = resolve_num_customers_from_config(config)
    total_demand = n_customers * (config.demand_min + config.demand_max) / 2
    return int(np.ceil(total_demand / max(1, num_sat) * 1.3))


def _stratified_satellite_select(
    candidates: List[Dict[str, Any]],
    *,
    anchor_lat: float,
    anchor_lon: float,
    num_select: int,
) -> List[Dict[str, Any]]:
    """
    Pick ``num_select`` transfer hubs without maximin corner-seeking.

    1. Sort OSM-snapped candidates by great-circle distance to the primary depot.
    2. Restrict to an **inner distance band** (10th–90th percentile by default) so
       sites are neither on top of the depot nor the single farthest fringe points
       (which often sit near the bbox / city corners).
    3. For ``k = 0 … num_select-1``, target bearings ``τ_k = -π + (k+0.5)·2π/num_select``
       around the depot; greedily assign the unused candidate in the band whose
       bearing is closest to ``τ_k``, tie-breaking by proximity to the band’s
       median distance (compact “ring” of similar drive distances).
    4. If the band is too small or ties exhaust uniqueness, fill from the full
       sorted list, then fall back to :func:`_farthest_first_select` on leftovers.
    """
    if num_select <= 0:
        return []
    for s in candidates:
        s["_d_anchor"] = _great_circle_m(s["lat"], s["lon"], anchor_lat, anchor_lon)
    ordered = sorted(candidates, key=lambda s: s["_d_anchor"])
    n = len(ordered)
    if n < num_select:
        for s in ordered:
            s.pop("_d_anchor", None)
        return []

    p_lo, p_hi = 0.10, 0.90
    lo = max(0, int(n * p_lo))
    hi = min(n, max(lo + num_select, int(math.ceil(n * p_hi))))
    band = ordered[lo:hi]
    if len(band) < num_select:
        band = list(ordered)

    d_vals = [float(s["_d_anchor"]) for s in band]
    d_med = float(np.median(d_vals)) if d_vals else 0.0

    for s in band:
        s["_bearing"] = _bearing_rad_from_anchor(
            anchor_lat, anchor_lon, float(s["lat"]), float(s["lon"]),
        )

    used: set = set()
    selected: List[Dict[str, Any]] = []

    for k in range(num_select):
        tau = -math.pi + (k + 0.5) * (2 * math.pi / num_select)
        best: Optional[Dict[str, Any]] = None
        best_key: Optional[Tuple[float, float, float]] = None
        for s in band:
            nid = int(s["movement_node_id"])
            if nid in used:
                continue
            bt = float(s["_bearing"])
            ang_err = abs((bt - tau + math.pi) % (2 * math.pi) - math.pi)
            dpen = abs(float(s["_d_anchor"]) - d_med)
            key = (ang_err, dpen, float(s["_d_anchor"]))
            if best is None or key < best_key:
                best, best_key = s, key
        if best is not None:
            selected.append(best)
            used.add(int(best["movement_node_id"]))

    if len(selected) < num_select:
        for s in ordered:
            if len(selected) >= num_select:
                break
            nid = int(s["movement_node_id"])
            if nid in used:
                continue
            selected.append(s)
            used.add(nid)

    if len(selected) < num_select:
        rest = [s for s in candidates if int(s["movement_node_id"]) not in used]
        ff = _farthest_first_select(
            rest,
            anchor_lat=anchor_lat,
            anchor_lon=anchor_lon,
            num_select=num_select - len(selected),
        )
        for s in ff:
            nid = int(s["movement_node_id"])
            if nid not in used:
                selected.append(s)
                used.add(nid)
            if len(selected) >= num_select:
                break

    for s in candidates:
        s.pop("_d_anchor", None)
        s.pop("_bearing", None)

    return selected[:num_select]


def _farthest_first_select(
    candidates: List[Dict[str, Any]],
    *,
    anchor_lat: float,
    anchor_lon: float,
    num_select: int,
) -> List[Dict[str, Any]]:
    """
    Deterministic maximin subset (legacy backfill for sparse candidate sets).

    Sort by distance to anchor, seed median, then repeatedly take the candidate
    maximizing the minimum distance to already chosen sites.
    """
    for s in candidates:
        s["_d_anchor"] = _great_circle_m(s["lat"], s["lon"], anchor_lat, anchor_lon)
    candidates = sorted(candidates, key=lambda s: s["_d_anchor"])
    if not candidates or num_select <= 0:
        return []
    selected: List[Dict[str, Any]] = [candidates[len(candidates) // 2]]
    used = {selected[0]["movement_node_id"]}
    while len(selected) < num_select:
        remaining = [s for s in candidates if s["movement_node_id"] not in used]
        if not remaining:
            break
        best = max(
            remaining,
            key=lambda s: min(
                _great_circle_m(s["lat"], s["lon"], o["lat"], o["lon"])
                for o in selected
            ),
        )
        selected.append(best)
        used.add(best["movement_node_id"])
    return selected


# ── Stage 5: customer generation ─────────────────────────────────────────

def generate_customers(state: PipelineState) -> PipelineState:
    """End-customer generation (shared with classic variant)."""
    csv_path = state.config.customer_csv_path
    if csv_path:
        imported = load_customers_from_csv(csv_path)
        return apply_customers_to_state(state, imported)
    return generate_customers_standard(state)


# ── Stage 6: customer → satellite assignment ─────────────────────────────

def assign_customers_to_satellites(state: PipelineState) -> PipelineState:
    """
    Second-echelon **customer ↔ satellite** coupling.

    Deterministic **nearest-hub** rule (Voronoi-style partition on the sphere metric
    used elsewhere in the library): each customer is assigned to the satellite that
    minimizes great-circle distance; ties break by smaller ``satellite.id``.

    Then each hub’s ``capacity`` is lifted to at least the **realized** assigned
    demand (keeping any higher preset from ``config.satellite_capacity`` / heuristic),
    matching a minimal **flow-feasible** second-echelon payload proxy for benchmarks.
    """
    satellites = state.satellites
    customers = state.customers

    if not satellites:
        return state

    assignment: Dict[int, List[int]] = {s.id: [] for s in satellites}
    cust_by_id = {c.id: c for c in customers}

    for c in customers:
        best = min(
            satellites,
            key=lambda s: (
                _great_circle_m(c.lat, c.lon, s.lat, s.lon),
                s.id,
            ),
        )
        assignment[best.id].append(c.id)

    for s in satellites:
        ids = assignment[s.id]
        s.assigned_customer_ids = ids
        realized = sum(cust_by_id[i].demand for i in ids)
        s.capacity = max(int(s.capacity), int(realized))

    state.satellites = satellites
    return state


# ── Stage 7: stations ─────────────────────────────────────────────────────

def generate_stations(state: PipelineState) -> PipelineState:
    """
    Station selection — unified extraction when available, else legacy path.

    Uses pre-snapped station candidates from ``run_unified_extraction_and_snap``
    when ``state._unified_extracted`` is true; otherwise Overpass + snap.
    """
    config = state.config

    if state._unified_extracted:
        real_candidates = list(state._unified_ev_stations) + list(state._unified_proxy_hosts)
        pre_synth = list(state._unified_synthetic_hosts)
    else:
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
        real_candidates = snap_stations(
            raw, state.movement_graph, config.real_stations_snap_max_dist_m,
        )
        pre_synth = None

    country_defaults = _country_defaults(config)
    blocked = {int(state.depot_node_id)} | {int(c.movement_node_id) for c in state.customers}
    for sat in state.satellites:
        blocked.add(int(sat.movement_node_id))
    state.stations = select_station_set(
        num_stations=config.num_stations,
        real_station_candidates=real_candidates,
        customers=state.customers,
        depot_lat=config.depot_lat, depot_lon=config.depot_lon,
        movement_graph=state.movement_graph, seed=config.seed,
        config=config, country_defaults=country_defaults,
        bbox=state.bbox, disk_cache=state.disk_cache,
        pre_snapped_synthetic=pre_synth,
        blocked_node_ids=blocked,
        repair_summary=state.repair_summary,
    )
    return state


# ── Stage 8–10: finalize ─────────────────────────────────────────────────

def finalize(
    state: PipelineState,
    *,
    compute_matrices: bool = True,
    run_energy_feasibility: bool = True,
) -> BenchmarkInstance:
    """Assemble the two-echelon BenchmarkInstance."""
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

        feasibility = build_two_echelon_report(
            movement_graph=G,
            config=config,
            ev_features=ev,
            depot_node_id=state.depot_node_id,
            customers=state.customers,
            stations=state.stations,
            satellites=state.satellites,
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
        feasibility = build_two_echelon_report(
            movement_graph=G,
            config=config,
            ev_features=ev,
            depot_node_id=state.depot_node_id,
            customers=state.customers,
            stations=state.stations,
            satellites=state.satellites,
            travel_time_matrix_s=tt_tmp[config.energy_period],
            energy_matrix_kwh=energy_tmp,
            depot_to_node_time=state.depot_to_node_time,
            compute_matrices=False,
            run_energy_feasibility=True,
            service_nodes=service_nodes,
        )
    else:
        feasibility = build_two_echelon_report(
            movement_graph=G,
            config=config,
            ev_features=ev,
            depot_node_id=state.depot_node_id,
            customers=state.customers,
            stations=state.stations,
            satellites=state.satellites,
            travel_time_matrix_s=None,
            energy_matrix_kwh=None,
            depot_to_node_time=state.depot_to_node_time,
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
        variant="two_echelon_evrp",
        time_window_tightness=config.time_window_tightness,
        feasibility_level=feas_level,
        depot_count=1,
        satellite_count=len(state.satellites),
        customer_count=len(state.customers),
        station_count_observed_ev=n_obs,
        station_count_proxy_host=n_proxy,
        station_count_synthetic=n_synth,
        elevation_enabled=(config.node_elevation_provider != "none"),
        two_echelon_enabled=True,
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
        satellites=state.satellites,
    )
    return attach_post_finalize_artifacts(instance, state.repair_summary)


# ── One-shot entry point ─────────────────────────────────────────────────

def generate_two_echelon_evrp(
    config: GenerationConfig,
    ev_features: Optional[EVFeatures] = None,
    movement_graph=None,
    compute_matrices: bool = True,
    run_energy_feasibility: bool = True,
) -> BenchmarkInstance:
    """Full two-echelon EVRP generation pipeline in one call."""
    if ev_features is None:
        ev_features = EVFeatures()

    state = prepare_graph_and_depot(config, ev_features, movement_graph)
    state = run_unified_extraction_and_snap(state)
    state = setup_satellites(state)
    state = generate_customers(state)
    state = assign_customers_to_satellites(state)
    state = generate_stations(state)
    return finalize(state, compute_matrices=compute_matrices, run_energy_feasibility=run_energy_feasibility)

"""Customer spatial selection: clustered and random placement, plus shared pipeline step."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..exceptions import EvrpUserError
from ..utils.snapping import snap_customers_to_graph
from ..utils.time_windows import TWProfile, assign_time_window, resolve_tw_profile
from ..types import (
    CustomerCandidate,
    CustomerRecord,
    GenerationConfig,
    GenerationRepairSummary,
    PipelineState,
)

from .extraction import extract_building_candidates
from .csv_import import resolve_num_customers_from_config

_EARTH_R = 6_371_000.0  # metres


def _haversine_vec(
    lat1: np.ndarray, lon1: np.ndarray,
    lat2: np.ndarray, lon2: np.ndarray,
) -> np.ndarray:
    """Vectorised haversine distance (metres). All inputs in degrees."""
    rlat1 = np.radians(lat1)
    rlat2 = np.radians(lat2)
    dlat = rlat2 - rlat1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat * 0.5) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon * 0.5) ** 2
    return _EARTH_R * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def _great_circle_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Scalar haversine — kept for small-N paths that aren't worth vectorising."""
    return float(
        _haversine_vec(
            np.array(lat1), np.array(lon1),
            np.array(lat2), np.array(lon2),
        )
    )


def _min_dist_to_picked(lat: float, lon: float, picked: List[Tuple[float, float]]) -> float:
    if not picked:
        return float("inf")
    arr = np.asarray(picked, dtype=np.float64)
    dists = _haversine_vec(
        np.full(len(arr), lat), np.full(len(arr), lon),
        arr[:, 0], arr[:, 1],
    )
    return float(dists.min())


def _eligible_for_radius(
    sorted_pairs: List[Tuple[float, CustomerCandidate]],
    max_radius_m: Optional[float],
    need_at_least: int,
) -> List[Tuple[float, CustomerCandidate]]:
    """
    Keep candidates within ``max_radius_m`` of their cluster center.
    Allow minor relaxation (up to 2x) only if the strict radius yields
    fewer than ``need_at_least`` candidates, but never beyond that.
    """
    if max_radius_m is None or max_radius_m <= 0:
        return list(sorted_pairs)

    target = min(max(1, need_at_least), len(sorted_pairs))
    R = float(max_radius_m)
    R_cap = R * 2.0
    while R <= R_cap:
        eligible = [(d, c) for d, c in sorted_pairs if d <= R]
        if len(eligible) >= target:
            return eligible
        R *= 1.3
    return [(d, c) for d, c in sorted_pairs if d <= R_cap]


def _pick_with_separation_then_fill(
    sorted_pairs: List[Tuple[float, CustomerCandidate]],
    quota: int,
    selected_ids: Set[int],
    picked_positions: List[Tuple[float, float]],
    min_sep_m: Optional[float],
    depot_travel_time_ref: Dict[int, float],
    tt_min: float,
    tt_max: float,
    config: GenerationConfig,
    tw_profile: TWProfile,
    rng: np.random.Generator,
    used_movement_nodes: Set[int],
    repair_summary: Optional[GenerationRepairSummary] = None,
) -> List[CustomerRecord]:
    """
    Greedily take closest-first, enforcing minimum separation when possible;
    then fill remaining quota without separation if needed.
    """
    sep = float(min_sep_m) if (min_sep_m is not None and min_sep_m > 0) else 0.0
    out: List[CustomerRecord] = []

    def try_take(cand: CustomerCandidate, enforce_sep: bool) -> bool:
        if cand.id in selected_ids:
            return False
        nid = int(cand.movement_node_id)
        if nid in used_movement_nodes:
            if repair_summary is not None:
                repair_summary.duplicate_rejections += 1
            return False
        if enforce_sep and sep > 0:
            if _min_dist_to_picked(cand.lat, cand.lon, picked_positions) < sep:
                return False
        t_ref = float(depot_travel_time_ref[cand.id])
        record = assign_time_window(
            candidate=cand,
            depot_travel_time_s=t_ref,
            depot_tt_min_s=tt_min,
            depot_tt_max_s=tt_max,
            config=config,
            tw_profile=tw_profile,
            rng=rng,
            repair_summary=repair_summary,
        )
        out.append(record)
        selected_ids.add(cand.id)
        used_movement_nodes.add(nid)
        picked_positions.append((cand.lat, cand.lon))
        return True

    # Pass 1 — respect separation
    if sep > 0:
        for _d, cand in sorted_pairs:
            if len(out) >= quota:
                break
            try_take(cand, enforce_sep=True)

    # Pass 2 — fill remainder without separation
    for _d, cand in sorted_pairs:
        if len(out) >= quota:
            break
        try_take(cand, enforce_sep=False)

    return out


def _global_top_up(
    candidates: List[CustomerCandidate],
    centers: List[CustomerCandidate],
    need: int,
    selected_ids: Set[int],
    picked_positions: List[Tuple[float, float]],
    min_sep_m: Optional[float],
    cluster_max_radius_m: Optional[float],
    depot_travel_time_ref: Dict[int, float],
    tt_min: float,
    tt_max: float,
    config: GenerationConfig,
    tw_profile: TWProfile,
    rng: np.random.Generator,
    used_movement_nodes: Set[int],
    repair_summary: Optional[GenerationRepairSummary] = None,
) -> List[CustomerRecord]:
    """
    Fill remaining slots from unused candidates that are still within the
    cluster zone (within ``cluster_max_radius_m`` of any center).
    """
    if need <= 0:
        return []
    remaining = [c for c in candidates if c.id not in selected_ids]
    if not remaining:
        return []

    max_r = float(cluster_max_radius_m) if cluster_max_radius_m and cluster_max_radius_m > 0 else float("inf")

    rem_lats = np.array([c.lat for c in remaining], dtype=np.float64)
    rem_lons = np.array([c.lon for c in remaining], dtype=np.float64)
    n_rem = len(remaining)

    center_dists = np.stack([
        _haversine_vec(
            np.full(n_rem, ce.lat), np.full(n_rem, ce.lon),
            rem_lats, rem_lons,
        )
        for ce in centers
    ])
    d_min_arr = center_dists.min(axis=0)

    scored: List[Tuple[float, CustomerCandidate]] = []
    for i, c in enumerate(remaining):
        if d_min_arr[i] <= max_r * 1.5:
            scored.append((float(d_min_arr[i]), c))
    scored.sort(key=lambda pair: (pair[0], pair[1].id))
    got = _pick_with_separation_then_fill(
        sorted_pairs=scored,
        quota=need,
        selected_ids=selected_ids,
        picked_positions=picked_positions,
        min_sep_m=min_sep_m,
        depot_travel_time_ref=depot_travel_time_ref,
        tt_min=tt_min,
        tt_max=tt_max,
        config=config,
        tw_profile=tw_profile,
        rng=rng,
        used_movement_nodes=used_movement_nodes,
        repair_summary=repair_summary,
    )
    if len(got) >= need:
        return got

    # Widen radius progressively (still near some cluster) before last resort.
    for mult in (2.5, 4.0, 8.0):
        need2 = need - len(got)
        if need2 <= 0:
            break
        scored2: List[Tuple[float, CustomerCandidate]] = []
        for i, c in enumerate(remaining):
            if c.id in selected_ids:
                continue
            if d_min_arr[i] <= max_r * mult:
                scored2.append((float(d_min_arr[i]), c))
        scored2.sort(key=lambda pair: (pair[0], pair[1].id))
        more = _pick_with_separation_then_fill(
            sorted_pairs=scored2,
            quota=need2,
            selected_ids=selected_ids,
            picked_positions=picked_positions,
            min_sep_m=None,
            depot_travel_time_ref=depot_travel_time_ref,
            tt_min=tt_min,
            tt_max=tt_max,
            config=config,
            tw_profile=tw_profile,
            rng=rng,
            used_movement_nodes=used_movement_nodes,
            repair_summary=repair_summary,
        )
        got.extend(more)
        if len(got) >= need:
            return got[:need]
    return got


def _last_resort_fill_clustered(
    candidates: List[CustomerCandidate],
    need: int,
    selected_ids: Set[int],
    depot_travel_time_ref: Dict[int, float],
    tt_min: float,
    tt_max: float,
    config: GenerationConfig,
    tw_profile: TWProfile,
    rng: np.random.Generator,
    used_movement_nodes: Set[int],
    repair_summary: Optional[GenerationRepairSummary] = None,
) -> List[CustomerRecord]:
    """
    Fill remaining slots with any reachable candidate (no cluster radius / no
    separation). Used only when stricter passes could not reach *need*.
    """
    if need <= 0:
        return []
    pool = [c for c in candidates if c.id not in selected_ids and c.id in depot_travel_time_ref]
    pool.sort(key=lambda c: (float(depot_travel_time_ref[c.id]), c.id))
    out: List[CustomerRecord] = []
    for cand in pool:
        if len(out) >= need:
            break
        nid = int(cand.movement_node_id)
        if nid in used_movement_nodes:
            if repair_summary is not None:
                repair_summary.duplicate_rejections += 1
            continue
        record = assign_time_window(
            candidate=cand,
            depot_travel_time_s=float(depot_travel_time_ref[cand.id]),
            depot_tt_min_s=tt_min,
            depot_tt_max_s=tt_max,
            config=config,
            tw_profile=tw_profile,
            rng=rng,
            repair_summary=repair_summary,
        )
        out.append(record)
        selected_ids.add(cand.id)
        used_movement_nodes.add(nid)
    return out


def generate_clustered_customers(
    num_customers: int,
    num_clusters: int,
    candidates: List[CustomerCandidate],
    depot_travel_time_ref: Dict[int, float],
    config: GenerationConfig,
    rng: np.random.Generator,
    used_movement_nodes: Set[int],
    repair_summary: Optional[GenerationRepairSummary] = None,
) -> List[CustomerRecord]:
    """
    Clustered customer generation (mode "c").

    Algorithm:
      1. Pre-filter candidates to a service zone around the depot centroid so
         clusters form in a compact area rather than spanning the whole city.
      2. Pick cluster centers via farthest-first (maximin) seeding within the
         filtered pool.
      3. Assign every candidate to its nearest center (Voronoi-style).
      4. Hard-filter each cluster to ``cluster_max_radius_m`` (minor relaxation
         up to 2x only).
      5. Fill each cluster quota with closest-first, enforcing
         ``cluster_min_separation_m`` when possible, then fill without it.
      6. Top up from remaining candidates **near** cluster centers only.
    """

    if num_customers <= 0:
        return []
    if len(candidates) < num_customers:
        raise ValueError("Not enough customer candidates")

    tw_profile = resolve_tw_profile(config)

    num_clusters = max(1, int(num_clusters))
    num_clusters = min(num_clusters, num_customers)

    all_tt = [float(depot_travel_time_ref[c.id]) for c in candidates if c.id in depot_travel_time_ref]
    tt_min = min(all_tt) if all_tt else 0.0
    tt_max = max(all_tt) if all_tt else 0.0

    # Build coordinate arrays once for vectorised distance computations
    _cand_lats = np.array([c.lat for c in candidates], dtype=np.float64)
    _cand_lons = np.array([c.lon for c in candidates], dtype=np.float64)
    n_cands = len(candidates)

    # ── 0) Pre-filter to a compact service zone ──────────────────────────
    max_r = config.cluster_max_radius_m
    if max_r is not None and max_r > 0:
        zone_radius_m = max_r * num_clusters * 1.2
        centroid_lat = float(_cand_lats.mean())
        centroid_lon = float(_cand_lons.mean())
        zone_dists = _haversine_vec(
            _cand_lats, _cand_lons,
            np.full(n_cands, centroid_lat), np.full(n_cands, centroid_lon),
        )
        mask = zone_dists <= zone_radius_m
        if int(mask.sum()) >= num_customers:
            candidates = [candidates[i] for i in range(n_cands) if mask[i]]
            _cand_lats = _cand_lats[mask]
            _cand_lons = _cand_lons[mask]
            n_cands = len(candidates)

    # ── 1) Farthest-first center selection (vectorised) ───────────────────
    centers: List[CustomerCandidate] = []
    center_ids = set()

    first_idx = int(rng.integers(0, n_cands))
    first = candidates[first_idx]
    centers.append(first)
    center_ids.add(first.id)

    dist_to_nearest = _haversine_vec(
        np.full(n_cands, first.lat), np.full(n_cands, first.lon),
        _cand_lats, _cand_lons,
    )
    is_center = np.zeros(n_cands, dtype=bool)
    is_center[first_idx] = True

    while len(centers) < num_clusters:
        masked = np.where(is_center, -1.0, dist_to_nearest)
        best_idx = int(np.argmax(masked))
        if masked[best_idx] <= 0:
            break
        new_center = candidates[best_idx]
        centers.append(new_center)
        center_ids.add(new_center.id)
        is_center[best_idx] = True
        d_new = _haversine_vec(
            np.full(n_cands, new_center.lat), np.full(n_cands, new_center.lon),
            _cand_lats, _cand_lons,
        )
        dist_to_nearest = np.minimum(dist_to_nearest, d_new)

    # ── 2) Assign every candidate to nearest center (vectorised) ──────────
    center_lats = np.array([ce.lat for ce in centers], dtype=np.float64)
    center_lons = np.array([ce.lon for ce in centers], dtype=np.float64)
    # dist_matrix shape: (n_centers, n_cands)
    dist_matrix = np.stack([
        _haversine_vec(
            np.full(n_cands, clat), np.full(n_cands, clon),
            _cand_lats, _cand_lons,
        )
        for clat, clon in zip(center_lats, center_lons)
    ])
    nearest_center = np.argmin(dist_matrix, axis=0)
    nearest_dist = dist_matrix[nearest_center, np.arange(n_cands)]

    cluster_groups: List[List[Tuple[float, CustomerCandidate]]] = [[] for _ in range(len(centers))]
    for i, cand in enumerate(candidates):
        cluster_groups[int(nearest_center[i])].append((float(nearest_dist[i]), cand))

    for group in cluster_groups:
        group.sort(key=lambda pair: pair[0])

    # ── 3) Quotas ─────────────────────────────────────────────────────────
    base_quota = num_customers // num_clusters
    remainder = num_customers - base_quota * num_clusters

    quotas = [base_quota] * len(centers)
    sizes = [(len(cluster_groups[k]), k) for k in range(len(centers))]
    sizes.sort(reverse=True)
    for i in range(remainder):
        quotas[sizes[i][1]] += 1

    selected_ids: Set[int] = set()
    picked_positions: List[Tuple[float, float]] = []
    final_customers: List[CustomerRecord] = []

    max_r = config.cluster_max_radius_m
    min_sep = config.cluster_min_separation_m

    for k in range(len(centers)):
        raw_group = cluster_groups[k]
        eligible = _eligible_for_radius(raw_group, max_r, quotas[k])
        chunk = _pick_with_separation_then_fill(
            sorted_pairs=eligible,
            quota=quotas[k],
            selected_ids=selected_ids,
            picked_positions=picked_positions,
            min_sep_m=min_sep,
            depot_travel_time_ref=depot_travel_time_ref,
            tt_min=tt_min,
            tt_max=tt_max,
            config=config,
            tw_profile=tw_profile,
            rng=rng,
            used_movement_nodes=used_movement_nodes,
            repair_summary=repair_summary,
        )
        if len(chunk) < quotas[k]:
            # Still short (e.g. heavy dedup): scan full Voronoi group
            extra = _pick_with_separation_then_fill(
                sorted_pairs=raw_group,
                quota=quotas[k] - len(chunk),
                selected_ids=selected_ids,
                picked_positions=picked_positions,
                min_sep_m=min_sep,
                depot_travel_time_ref=depot_travel_time_ref,
                tt_min=tt_min,
                tt_max=tt_max,
                config=config,
                tw_profile=tw_profile,
                rng=rng,
                used_movement_nodes=used_movement_nodes,
                repair_summary=repair_summary,
            )
            chunk.extend(extra)
        final_customers.extend(chunk)

    if len(final_customers) < num_customers:
        need = num_customers - len(final_customers)
        top_up = _global_top_up(
            candidates=candidates,
            centers=centers,
            need=need,
            selected_ids=selected_ids,
            picked_positions=picked_positions,
            min_sep_m=min_sep,
            cluster_max_radius_m=max_r,
            depot_travel_time_ref=depot_travel_time_ref,
            tt_min=tt_min,
            tt_max=tt_max,
            config=config,
            tw_profile=tw_profile,
            rng=rng,
            used_movement_nodes=used_movement_nodes,
            repair_summary=repair_summary,
        )
        final_customers.extend(top_up)

    if len(final_customers) < num_customers:
        need_lr = num_customers - len(final_customers)
        lr = _last_resort_fill_clustered(
            candidates=candidates,
            need=need_lr,
            selected_ids=selected_ids,
            depot_travel_time_ref=depot_travel_time_ref,
            tt_min=tt_min,
            tt_max=tt_max,
            config=config,
            tw_profile=tw_profile,
            rng=rng,
            used_movement_nodes=used_movement_nodes,
            repair_summary=repair_summary,
        )
        final_customers.extend(lr)

    if len(final_customers) < num_customers:
        raise ValueError(
            f"Clustered selection could only place {len(final_customers)} customers "
            f"(requested {num_customers}) — not enough reachable building candidates. "
            f"Try raising customers_pool_snap_max_dist_m, lowering num_customers, "
            f"or reducing num_clusters / cluster_min_separation_m."
        )

    return final_customers[:num_customers]


def generate_random_customers(
    num_customers: int,
    candidates: List[CustomerCandidate],
    depot_travel_time_ref: Dict[int, float],
    config: GenerationConfig,
    rng: np.random.Generator,
    used_movement_nodes: Set[int],
    repair_summary: Optional[GenerationRepairSummary] = None,
) -> List[CustomerRecord]:
    """
    Farthest-point sampling (Gonzalez 1985) — maximally dispersed selection.

    Each new customer is the candidate farthest from all previously selected
    customers.  This is a 2-approximation for the k-center dispersion
    objective and produces spatially diverse layouts without the bookkeeping
    of the old active/backup pool mechanism.

    Complexity: O(n * N) — n iterations of O(N) vectorised numpy ops.
    """
    if num_customers <= 0:
        return []
    if len(candidates) < num_customers:
        raise ValueError("Not enough customer candidates")

    tw_profile = resolve_tw_profile(config)

    all_tt = [float(depot_travel_time_ref[c.id]) for c in candidates if c.id in depot_travel_time_ref]
    tt_min = min(all_tt) if all_tt else 0.0
    tt_max = max(all_tt) if all_tt else 0.0

    N = len(candidates)
    X_lat = np.array([c.lat for c in candidates], dtype=np.float64)
    X_lon = np.array([c.lon for c in candidates], dtype=np.float64)

    first_idx = int(rng.integers(0, N))
    selected_indices: List[int] = []
    for attempt in range(N):
        idx = (first_idx + attempt) % N
        if int(candidates[idx].movement_node_id) in used_movement_nodes:
            continue
        selected_indices.append(idx)
        break
    if not selected_indices:
        raise ValueError("No customer candidate with unused movement node.")

    d_min = _haversine_vec(
        np.full(N, X_lat[selected_indices[0]]), np.full(N, X_lon[selected_indices[0]]),
        X_lat, X_lon,
    )
    d_min[selected_indices[0]] = -1.0

    while len(selected_indices) < num_customers:
        next_idx = int(np.argmax(d_min))
        if d_min[next_idx] <= 0:
            break
        if int(candidates[next_idx].movement_node_id) in used_movement_nodes:
            d_min[next_idx] = -1.0
            if repair_summary is not None:
                repair_summary.duplicate_rejections += 1
            continue
        selected_indices.append(next_idx)
        d_new = _haversine_vec(
            np.full(N, X_lat[next_idx]), np.full(N, X_lon[next_idx]),
            X_lat, X_lon,
        )
        d_min = np.minimum(d_min, d_new)
        d_min[next_idx] = -1.0

    final_customers: List[CustomerRecord] = []
    for idx in selected_indices:
        cand = candidates[idx]
        record = assign_time_window(
            candidate=cand,
            depot_travel_time_s=float(depot_travel_time_ref[cand.id]),
            depot_tt_min_s=tt_min,
            depot_tt_max_s=tt_max,
            config=config,
            tw_profile=tw_profile,
            rng=rng,
            repair_summary=repair_summary,
        )
        final_customers.append(record)
        used_movement_nodes.add(int(cand.movement_node_id))

    if len(final_customers) < num_customers:
        raise ValueError(
            f"Random selection could only place {len(final_customers)} customers "
            f"(requested {num_customers}) with unique road nodes."
        )

    return final_customers


def assign_customer_attributes(
    candidate: CustomerCandidate,
    depot_travel_time_s: float,
    depot_travel_times_min_s: float,
    depot_travel_times_max_s: float,
    config: GenerationConfig,
    rng: np.random.Generator,
    tw_profile: Optional[TWProfile] = None,
) -> CustomerRecord:
    if tw_profile is None:
        tw_profile = resolve_tw_profile(config)
    return assign_time_window(
        candidate,
        depot_travel_time_s,
        depot_travel_times_min_s,
        depot_travel_times_max_s,
        config,
        tw_profile,
        rng,
    )


def generate_customers_standard(state: PipelineState) -> PipelineState:
    """
    Shared customer generation for classic, multi-depot, and two-echelon variants.

    Uses pre-extracted buildings from unified extraction (Algorithm 1) when
    available, otherwise falls back to the legacy per-module extraction.
    """
    config = state.config
    n_customers = resolve_num_customers_from_config(config)

    # Use unified pre-extracted buildings if available (Algorithm 2 ran first)
    if state._unified_extracted and state._unified_buildings:
        snapped = list(state._unified_buildings)
    else:
        min_cand_mult = 3 if config.customer_pattern in ("c", "rc") else 2
        default_building = max(80, n_customers * min_cand_mult)
        building_min = (
            max(int(config.customer_building_osm_min_candidates), n_customers)
            if config.customer_building_osm_min_candidates is not None
            else default_building
        )
        raw = extract_building_candidates(
            config.city, config.country,
            bbox=state.bbox, min_candidates=building_min,
            disk_cache=state.disk_cache,
        )
        snapped = snap_customers_to_graph(raw, state.movement_graph, config.customers_pool_snap_max_dist_m)

    candidate_tt: Dict[int, float] = {}
    filtered = []
    for c in snapped:
        t = state.depot_to_node_time.get(c.movement_node_id)
        if t is None:
            continue
        filtered.append(c)
        candidate_tt[c.id] = float(t) / state.ev_features.speed_multiplier

    if len(filtered) < n_customers:
        raise EvrpUserError(
            f"Not enough reachable building candidates: got {len(filtered)}, "
            f"need {n_customers}."
        )

    used_nodes: Set[int] = {int(state.depot_node_id)}

    if config.customer_pattern == "c":
        customers = generate_clustered_customers(
            n_customers, config.num_clusters,
            filtered, candidate_tt, config, state.rng,
            used_movement_nodes=used_nodes,
            repair_summary=state.repair_summary,
        )
    elif config.customer_pattern == "r":
        customers = generate_random_customers(
            n_customers, filtered, candidate_tt, config, state.rng,
            used_movement_nodes=used_nodes,
            repair_summary=state.repair_summary,
        )
    elif config.customer_pattern == "rc":
        n_c = n_customers // 2
        n_r = n_customers - n_c
        clustered = generate_clustered_customers(
            n_c, config.num_clusters, list(filtered), candidate_tt, config, state.rng,
            used_movement_nodes=used_nodes,
            repair_summary=state.repair_summary,
        )
        used = {c.id for c in clustered}
        rest = [c for c in filtered if c.id not in used]
        random_part = generate_random_customers(
            n_r, rest, candidate_tt, config, state.rng,
            used_movement_nodes=used_nodes,
            repair_summary=state.repair_summary,
        )
        customers = clustered + random_part
    else:
        raise EvrpUserError(f"Unknown customer_pattern {config.customer_pattern!r}.")

    state.customers = customers
    return state

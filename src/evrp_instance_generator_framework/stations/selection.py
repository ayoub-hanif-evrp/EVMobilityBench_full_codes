"""Station selection, synthetic hosts, and attribute assignment."""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..exceptions import EvrpUserError
from ..types import CustomerRecord, GenerationConfig, GenerationRepairSummary, StationCandidate, StationRecord, StationType
from ..utils.snapping import snap_candidates_batch


def build_synthetic_station_hosts(
    movement_graph,
    config: GenerationConfig,
    rng: np.random.Generator,
    max_hosts: int = 200,
    max_snap_dist_m: Optional[float] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    disk_cache=None,
) -> List[Dict[str, Any]]:
    """
    Build synthetic station host candidates from OSM land-use features.

    Uses cached Overpass queries and batch-snapping for speed.
    All returned dicts carry ``station_source_type="synthetic"`` and
    ``source_priority=3``.
    """
    from ..data.osm_disk_cache import overpass_features_cached

    if max_snap_dist_m is None:
        max_snap_dist_m = config.real_stations_snap_max_dist_m

    # Single combined query covers parking + commercial + services
    tag_sets: List[Dict[str, Any]] = [
        {"amenity": ["parking", "service_area"], "landuse": "commercial", "highway": "services"},
    ]

    raw_points: List[Dict[str, Any]] = []
    seen: set = set()

    for tags in tag_sets:
        gdf = overpass_features_cached(
            tags, bbox, config.city, config.country, disk_cache,
            namespace="synthetic_hosts",
        )
        if gdf is None or len(gdf) == 0:
            continue
        gdf = gdf.reset_index(drop=True)
        for geom in gdf.geometry:
            if geom is None:
                continue
            c = geom.centroid if hasattr(geom, "centroid") else geom
            if not (hasattr(c, "y") and hasattr(c, "x")):
                continue
            lat, lon = float(c.y), float(c.x)
            key = (round(lat, 6), round(lon, 6))
            if key in seen:
                continue
            seen.add(key)
            raw_points.append({"id": len(raw_points), "lat": lat, "lon": lon})

    if not raw_points:
        return []

    snapped = snap_candidates_batch(raw_points, movement_graph, float(max_snap_dist_m))

    hosts: List[Dict[str, Any]] = []
    for s in snapped[:max_hosts]:
        host_type = "fast" if float(rng.random()) < config.station_fast_fraction else "normal"
        hosts.append({
            "lat": s["lat"],
            "lon": s["lon"],
            "movement_node_id": s["movement_node_id"],
            "snap_distance_m": s["snap_distance_m"],
            "host_type": host_type,
            "source": "synthetic",
            "station_source_type": "synthetic",
            "source_priority": 3,
            "is_real_observed_ev": False,
            "osm_tags": None,
        })

    return hosts


# ---------------------------------------------------------------------------
# Station attributes → StationRecord
# ---------------------------------------------------------------------------

_REQUIRED_STATION_DEFAULT_KEYS = (
    "price_fast_per_kWh",
    "price_normal_per_kWh",
    "green_carbon_intensity",
    "non_green_carbon_intensity",
    "slot_defaults",
)


def _require_resolved_default(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(
            f"Resolved country/city defaults are missing '{key}'. "
            f"Define it under 'default' in countries_data.json."
        )
    return d[key]


def _assign_station_type(
    station: Dict[str, Any],
    config: GenerationConfig,
    rng: np.random.Generator,
) -> StationType:
    if "host_type" in station and station["host_type"] in {"fast", "normal"}:
        return station["host_type"]

    power_hint = station.get("charging_power_kW_hint")
    if power_hint is not None:
        threshold = 0.75 * config.fast_charging_power_kW
        return "fast" if float(power_hint) >= threshold else "normal"

    return "fast" if float(rng.random()) < config.station_fast_fraction else "normal"


def assign_station_attributes(
    station_locations: List[Dict[str, Any]],
    config: GenerationConfig,
    country_defaults: Dict[str, Any],
    rng: np.random.Generator,
) -> List[StationRecord]:
    """
    Assign static dataset attributes and provenance to station locations.

    Each input dict may carry provenance keys
    (``station_source_type``, ``source_priority``, ``is_real_observed_ev``,
    ``osm_tags``) set during extraction.  These are copied into the final
    ``StationRecord``.
    """
    for key in _REQUIRED_STATION_DEFAULT_KEYS:
        _require_resolved_default(country_defaults, key)

    price_fast = float(country_defaults["price_fast_per_kWh"])
    price_normal = float(country_defaults["price_normal_per_kWh"])
    _carbon_green = float(country_defaults["green_carbon_intensity"])
    _carbon_non_green = float(country_defaults["non_green_carbon_intensity"])
    _ = (_carbon_green, _carbon_non_green)

    slot_defaults = country_defaults["slot_defaults"]
    if not isinstance(slot_defaults, dict):
        raise TypeError("country_defaults['slot_defaults'] must be a JSON object.")
    for st_key in ("fast", "normal"):
        if st_key not in slot_defaults:
            raise KeyError(f"slot_defaults must include '{st_key}'.")

    time_open_s = int(config.depot_time_open_s)
    time_close_s = int(config.depot_time_close_s)

    out: List[StationRecord] = []
    for idx, st in enumerate(station_locations):
        station_type = _assign_station_type(st, config=config, rng=rng)
        charging_power_kW = float(
            st.get("charging_power_kW_hint")
            if st.get("charging_power_kW_hint") is not None
            else (config.fast_charging_power_kW if station_type == "fast" else config.normal_charging_power_kW)
        )

        num_slots_hint = st.get("num_slots_hint")
        default_slots = int(slot_defaults[station_type])
        number_slots = int(max(config.default_min_slots, num_slots_hint if num_slots_hint is not None else default_slots))

        charging_price_per_kWh = float(price_fast if station_type == "fast" else price_normal)

        is_green_hint = st.get("is_green_hint")
        if is_green_hint in (0, 1):
            green_label = int(is_green_hint)
        else:
            green_label = 1 if float(rng.random()) < config.green_station_fraction else 0

        # Provenance (propagated from extraction / synthetic host generation)
        src_type = st.get("station_source_type", "synthetic")
        src_prio = st.get("source_priority", 3)
        is_real = st.get("is_real_observed_ev", False)
        raw_tags = st.get("osm_tags")

        # Backward-compat ``source`` field: "real" for observed_ev/proxy_host, "synthetic" otherwise
        legacy_source = str(st.get("source", "real" if src_prio <= 2 else "synthetic"))

        out.append(
            StationRecord(
                id=int(idx),
                lat=float(st["lat"]),
                lon=float(st["lon"]),
                movement_node_id=int(st["movement_node_id"]),
                snap_distance_m=float(st.get("snap_distance_m", 0.0)),
                time_open_s=time_open_s,
                time_close_s=time_close_s,
                number_slots=number_slots,
                station_type=station_type,
                charging_power_kW=charging_power_kW,
                charging_price_per_kWh=charging_price_per_kWh,
                green_label=green_label,
                source=legacy_source,
                station_source_type=src_type,
                source_priority=src_prio,
                is_real_observed_ev=is_real,
                osm_tags=raw_tags,
            )
        )

    return out


# ---------------------------------------------------------------------------
# Station set selection (k-medoids + synthetic greedy fill)
# ---------------------------------------------------------------------------


_EARTH_R = 6_371_000.0


def _haversine_vec(lat1, lon1, lat2, lon2):
    """Vectorised haversine (metres). All inputs numpy arrays in degrees."""
    rlat1, rlat2 = np.radians(lat1), np.radians(lat2)
    dlat, dlon = rlat2 - rlat1, np.radians(lon2 - lon1)
    a = np.sin(dlat * 0.5) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon * 0.5) ** 2
    return _EARTH_R * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def select_real_stations_by_kmedoids(
    candidates: List[StationCandidate],
    k: int,
    seed: int,
    max_iter: int = 10,
) -> List[StationCandidate]:
    """
    Vectorised k-medoids (Algorithm 5a).

    Farthest-first initialisation → Lloyd refinement (≤ max_iter rounds).
    All distance computations use numpy matrix ops.
    """
    if k <= 0:
        return []
    if len(candidates) <= k:
        return list(candidates)

    P = len(candidates)
    X_lat = np.array([c.lat for c in candidates], dtype=np.float64)
    X_lon = np.array([c.lon for c in candidates], dtype=np.float64)

    rng = np.random.default_rng(seed)

    # Farthest-first initialisation
    M = [int(rng.integers(0, P))]
    d_near = _haversine_vec(
        np.full(P, X_lat[M[0]]), np.full(P, X_lon[M[0]]),
        X_lat, X_lon,
    )
    d_near[M[0]] = -1.0

    for _ in range(1, k):
        next_idx = int(np.argmax(d_near))
        if d_near[next_idx] <= 0:
            break
        M.append(next_idx)
        d_new = _haversine_vec(
            np.full(P, X_lat[next_idx]), np.full(P, X_lon[next_idx]),
            X_lat, X_lon,
        )
        d_near = np.minimum(d_near, d_new)
        d_near[next_idx] = -1.0

    # Lloyd refinement
    for _ in range(max_iter):
        m_lat = X_lat[M]
        m_lon = X_lon[M]
        D = np.stack([
            _haversine_vec(np.full(P, ml), np.full(P, mo), X_lat, X_lon)
            for ml, mo in zip(m_lat, m_lon)
        ])  # shape (k_actual, P)
        labels = np.argmin(D, axis=0)

        M_new = []
        for j in range(len(M)):
            mask = labels == j
            if not mask.any():
                M_new.append(M[j])
                continue
            cluster_D = D[j, mask]
            within = np.where(mask)[0]
            sum_d = np.zeros(len(within))
            for jj in range(len(within)):
                sum_d[jj] = _haversine_vec(
                    np.full(mask.sum(), X_lat[within[jj]]),
                    np.full(mask.sum(), X_lon[within[jj]]),
                    X_lat[mask], X_lon[mask],
                ).sum()
            M_new.append(int(within[np.argmin(sum_d)]))

        if set(M_new) == set(M):
            break
        M = M_new

    # Ensure we have exactly k unique medoids
    selected_set = set(M)
    result = M[:k]
    if len(result) < k:
        remaining = [i for i in range(P) if i not in selected_set]
        d_near2 = np.full(P, np.inf)
        for m in result:
            d_new = _haversine_vec(np.full(P, X_lat[m]), np.full(P, X_lon[m]), X_lat, X_lon)
            d_near2 = np.minimum(d_near2, d_new)
        for m in result:
            d_near2[m] = -1.0
        while len(result) < k and remaining:
            best = max(remaining, key=lambda i: d_near2[i])
            result.append(best)
            d_near2[best] = -1.0
            remaining.remove(best)

    return [candidates[i] for i in result[:k]]


def _to_location_dict(c: StationCandidate) -> Dict[str, Any]:
    """Convert a typed candidate to a mutable dict for attribute assignment."""
    return {
        "movement_node_id": c.movement_node_id,
        "lat": c.lat,
        "lon": c.lon,
        "snap_distance_m": c.snap_distance_m,
        "source": "real" if c.source_priority <= 2 else "synthetic",
        "is_green_hint": c.is_green_hint,
        "charging_power_kW_hint": c.charging_power_kW_hint,
        "num_slots_hint": c.num_slots_hint,
        "station_source_type": c.station_source_type,
        "source_priority": c.source_priority,
        "is_real_observed_ev": (c.station_source_type == "observed_ev"),
        "osm_tags": c.osm_tags,
    }


def _customer_zone(
    customers: List[CustomerRecord],
    depot_lat: float,
    depot_lon: float,
) -> Tuple[float, float, float]:
    """Return (centroid_lat, centroid_lon, radius_m) of the customer zone."""
    c_lats = np.array([c.lat for c in customers] + [depot_lat], dtype=np.float64)
    c_lons = np.array([c.lon for c in customers] + [depot_lon], dtype=np.float64)
    clat = float(c_lats.mean())
    clon = float(c_lons.mean())
    dists = _haversine_vec(np.full(len(c_lats), clat), np.full(len(c_lats), clon), c_lats, c_lons)
    radius = float(dists.max()) * 1.3 + 500.0
    return clat, clon, radius


def _filter_by_customer_zone(
    candidates: List[StationCandidate],
    zone_lat: float,
    zone_lon: float,
    zone_radius_m: float,
) -> List[StationCandidate]:
    """Keep only candidates within the customer zone radius."""
    if not candidates:
        return []
    c_lats = np.array([c.lat for c in candidates], dtype=np.float64)
    c_lons = np.array([c.lon for c in candidates], dtype=np.float64)
    dists = _haversine_vec(
        np.full(len(c_lats), zone_lat), np.full(len(c_lats), zone_lon),
        c_lats, c_lons,
    )
    return [candidates[i] for i in range(len(candidates)) if dists[i] <= zone_radius_m]


def _filter_blocked_candidates(
    candidates: List[StationCandidate],
    blocked: Set[int],
    repair_summary: Optional[GenerationRepairSummary] = None,
) -> List[StationCandidate]:
    out: List[StationCandidate] = []
    for c in candidates:
        nid = int(c.movement_node_id)
        if nid in blocked:
            if repair_summary is not None:
                repair_summary.station_resamples += 1
            continue
        out.append(c)
    return out


def _append_unique_station(
    station_locations: List[Dict[str, Any]],
    used_nodes: Set[int],
    loc: Dict[str, Any],
    repair_summary: Optional[GenerationRepairSummary] = None,
) -> bool:
    nid = int(loc["movement_node_id"])
    if nid in used_nodes:
        if repair_summary is not None:
            repair_summary.station_resamples += 1
        return False
    station_locations.append(loc)
    used_nodes.add(nid)
    return True


def select_station_set(
    num_stations: int,
    real_station_candidates: List[StationCandidate],
    customers: List[CustomerRecord],
    depot_lat: float,
    depot_lon: float,
    movement_graph,
    seed: int,
    config: GenerationConfig,
    country_defaults: Dict[str, Any],
    bbox: Optional[Tuple[float, float, float, float]] = None,
    disk_cache=None,
    pre_snapped_synthetic: Optional[List[Dict[str, Any]]] = None,
    blocked_node_ids: Optional[Set[int]] = None,
    repair_summary: Optional[GenerationRepairSummary] = None,
) -> List[StationRecord]:
    """
    Select *num_stations* from candidates, honoring source priority.

    When ``pre_snapped_synthetic`` is provided (from unified extraction),
    no additional Overpass queries are made for synthetic hosts.
    """
    if num_stations <= 0:
        return []

    rng = np.random.default_rng(seed)
    blocked = set(int(n) for n in (blocked_node_ids or set()))
    used_nodes: Set[int] = set(blocked)

    if customers:
        z_lat, z_lon, z_r = _customer_zone(customers, depot_lat, depot_lon)
        zone_filtered = _filter_by_customer_zone(real_station_candidates, z_lat, z_lon, z_r)
        if len(zone_filtered) >= num_stations:
            real_station_candidates = zone_filtered

    real_station_candidates = _filter_blocked_candidates(
        real_station_candidates, blocked, repair_summary
    )

    p1 = [c for c in real_station_candidates if getattr(c, "source_priority", 1) == 1]
    p2 = [c for c in real_station_candidates if getattr(c, "source_priority", 2) == 2]

    station_locations: List[Dict[str, Any]] = []
    remaining_need = num_stations

    if p1:
        selected = select_real_stations_by_kmedoids(p1, min(remaining_need, len(p1)), seed)
        for c in selected:
            if remaining_need <= 0:
                break
            if _append_unique_station(
                station_locations, used_nodes, _to_location_dict(c), repair_summary
            ):
                remaining_need -= 1

    if remaining_need > 0 and p2:
        selected = select_real_stations_by_kmedoids(p2, min(remaining_need, len(p2)), seed + 1)
        for c in selected:
            if remaining_need <= 0:
                break
            if _append_unique_station(
                station_locations, used_nodes, _to_location_dict(c), repair_summary
            ):
                remaining_need -= 1

    # Priority 3: synthetic fill with vectorised coverage gain
    if remaining_need > 0:
        if not customers:
            raise EvrpUserError(
                "Station selection cannot add synthetic chargers without customers. "
                "Generate customers first, then select stations."
            )
        existing_nodes = set(used_nodes)

        if pre_snapped_synthetic is not None:
            synthetic_hosts = [
                {**h, "host_type": "fast" if float(rng.random()) < config.station_fast_fraction else "normal"}
                for h in pre_snapped_synthetic
                if int(h["movement_node_id"]) not in existing_nodes
            ]
        else:
            synthetic_hosts = build_synthetic_station_hosts(
                movement_graph=movement_graph, config=config, rng=rng, bbox=bbox,
                disk_cache=disk_cache,
            )
            synthetic_hosts = [
                h for h in synthetic_hosts if int(h["movement_node_id"]) not in existing_nodes
            ]

        zone_radius = None
        if customers:
            z_lat, z_lon, zone_radius = _customer_zone(customers, depot_lat, depot_lon)
            h_lats = np.array([h["lat"] for h in synthetic_hosts])
            h_lons = np.array([h["lon"] for h in synthetic_hosts])
            if len(h_lats) > 0:
                zone_dists = _haversine_vec(
                    np.full(len(h_lats), z_lat), np.full(len(h_lats), z_lon),
                    h_lats, h_lons,
                )
                zone_mask = zone_dists <= zone_radius
                synthetic_hosts = [synthetic_hosts[i] for i in range(len(synthetic_hosts)) if zone_mask[i]]

        cust_lats = np.array([c.lat for c in customers], dtype=np.float64)
        cust_lons = np.array([c.lon for c in customers], dtype=np.float64)
        depot_weight = float(config.depot_weight_for_station_coverage)

        zone_expand = 1.0
        while remaining_need > 0:
            if not synthetic_hosts:
                if zone_radius is not None and zone_expand < 3.0:
                    zone_expand *= 1.5
                    if pre_snapped_synthetic is not None:
                        pool = pre_snapped_synthetic
                    else:
                        pool = build_synthetic_station_hosts(
                            movement_graph=movement_graph, config=config, rng=rng, bbox=bbox,
                            disk_cache=disk_cache, max_hosts=400,
                        )
                    synthetic_hosts = [
                        {**h, "host_type": "fast" if float(rng.random()) < config.station_fast_fraction else "normal"}
                        for h in pool
                        if int(h["movement_node_id"]) not in used_nodes
                    ]
                    if customers and zone_radius is not None:
                        z_lat, z_lon, z_r = _customer_zone(customers, depot_lat, depot_lon)
                        expanded_r = z_r * zone_expand
                        h_lats = np.array([h["lat"] for h in synthetic_hosts])
                        h_lons = np.array([h["lon"] for h in synthetic_hosts])
                        if len(h_lats) > 0:
                            zone_dists = _haversine_vec(
                                np.full(len(h_lats), z_lat), np.full(len(h_lats), z_lon),
                                h_lats, h_lons,
                            )
                            synthetic_hosts = [
                                synthetic_hosts[i]
                                for i in range(len(synthetic_hosts))
                                if zone_dists[i] <= expanded_r
                            ]
                    continue
                raise EvrpUserError(
                    f"Could not select {num_stations} unique station road nodes "
                    f"(got {len(station_locations)}). Broaden the city area or reduce num_stations."
                )

            h_lats = np.array([h["lat"] for h in synthetic_hosts], dtype=np.float64)
            h_lons = np.array([h["lon"] for h in synthetic_hosts], dtype=np.float64)

            if station_locations:
                s_lats = np.array([s["lat"] for s in station_locations], dtype=np.float64)
                s_lons = np.array([s["lon"] for s in station_locations], dtype=np.float64)
            else:
                s_lats = None
                s_lons = None

            gains = _vectorized_coverage_gain(
                h_lats, h_lons, cust_lats, cust_lons,
                s_lats, s_lons, depot_lat, depot_lon, depot_weight,
            )
            best_idx = int(np.argmax(gains))
            best_host = synthetic_hosts.pop(best_idx)

            added = _append_unique_station(
                station_locations,
                used_nodes,
                {
                    "movement_node_id": best_host["movement_node_id"],
                    "lat": best_host["lat"],
                    "lon": best_host["lon"],
                    "snap_distance_m": best_host.get("snap_distance_m", 0.0),
                    "source": "synthetic",
                    "host_type": best_host.get("host_type", "normal"),
                    "is_green_hint": None,
                    "charging_power_kW_hint": None,
                    "num_slots_hint": None,
                    "station_source_type": "synthetic",
                    "source_priority": 3,
                    "is_real_observed_ev": False,
                    "osm_tags": None,
                },
                repair_summary,
            )
            if added:
                remaining_need -= 1

    if len(station_locations) != num_stations:
        raise EvrpUserError(
            f"Station selection produced {len(station_locations)} stations, expected {num_stations}."
        )

    return assign_station_attributes(
        station_locations=station_locations,
        config=config,
        country_defaults=country_defaults,
        rng=rng,
    )


def _vectorized_coverage_gain(
    host_lats: np.ndarray,
    host_lons: np.ndarray,
    cust_lats: np.ndarray,
    cust_lons: np.ndarray,
    existing_station_lats: Optional[np.ndarray],
    existing_station_lons: Optional[np.ndarray],
    depot_lat: float,
    depot_lon: float,
    depot_weight: float,
) -> np.ndarray:
    """
    Vectorised coverage gain for ALL candidate hosts at once (Algorithm 5 Phase 3).

    Returns shape (H,) — one score per host.
    """
    H = len(host_lats)
    C = len(cust_lats)
    EPS = 1e-9

    # D_host: shape (C, H) — distance from each customer to each host
    D_host = np.stack([
        _haversine_vec(
            np.full(C, host_lats[h]), np.full(C, host_lons[h]),
            cust_lats, cust_lons,
        )
        for h in range(H)
    ], axis=1)

    if existing_station_lats is not None and len(existing_station_lats) > 0:
        S = len(existing_station_lats)
        # D_cur: shape (C,) — min distance from each customer to nearest existing station
        D_exist = np.stack([
            _haversine_vec(
                np.full(C, existing_station_lats[s]),
                np.full(C, existing_station_lons[s]),
                cust_lats, cust_lons,
            )
            for s in range(S)
        ], axis=1)  # shape (C, S)
        D_cur = D_exist.min(axis=1)  # shape (C,)

        gain = np.maximum(0.0, D_cur[:, None] - D_host).sum(axis=0)  # shape (H,)

        # Depot term
        d_depot_exist = _haversine_vec(
            np.full(S, depot_lat), np.full(S, depot_lon),
            existing_station_lats, existing_station_lons,
        ).min()
        d_depot_hosts = _haversine_vec(
            np.full(H, depot_lat), np.full(H, depot_lon),
            host_lats, host_lons,
        )
        gain += depot_weight * np.maximum(0.0, d_depot_exist - d_depot_hosts)
    else:
        gain = (1.0 / (EPS + D_host)).sum(axis=0)  # shape (H,)
        d_depot_hosts = _haversine_vec(
            np.full(H, depot_lat), np.full(H, depot_lon),
            host_lats, host_lons,
        )
        gain += depot_weight / (EPS + d_depot_hosts)

    return gain

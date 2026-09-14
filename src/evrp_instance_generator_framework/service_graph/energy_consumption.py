from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from ..types import EVFeatures, Period


def compute_energy_matrix(
    movement_graph: nx.DiGraph,
    service_nodes: List[int],
    period: Period,
    ev_features: EVFeatures,
    precomputed_travel_time: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute arc energy on the service graph as the sum of segment energies
    along the shortest (by travel time) directed road path.

    Implements the Zhang & Yao tractive-force model:
    - rolling resistance
    - slope force (elevation)
    - aerodynamic drag
    - acceleration (finite-difference dv'/dt between consecutive edges)
    - HVAC + auxiliary power (Donkers et al.)

    Performance: uses one ``single_source_dijkstra`` per source node (returns
    both distances and full shortest-path trees), giving all N targets per call.
    Total: N Dijkstra runs instead of N² individual shortest_path calls.
    """

    n = len(service_nodes)
    energy_kwh = np.zeros((n, n), dtype=float)
    weight_attr = f"{period}_travel_time_s"
    node_to_idx = {nid: i for i, nid in enumerate(service_nodes)}

    # ── EV physics constants ─────────────────────────────────────────────────
    g = 9.81
    f = float(ev_features.rolling_resistance_coeff_f)
    m = float(ev_features.mass_kg)
    delta = float(ev_features.mass_factor_delta)
    cd = float(ev_features.drag_coefficient_cd)
    av = float(ev_features.frontal_area_m2)
    rho = float(ev_features.air_density_kg_m3)
    s_mult = ev_features.speed_multiplier

    h = int(ev_features.heating_on)
    c = int(ev_features.cooling_on)
    k_rain = int(ev_features.raining_on)

    p_hvac = 2000.0 * h + 1000.0 * c + 1000.0
    p_aux = 76.0 * h + 95.0 * (1 - h) + 60.0 * k_rain + 60.0

    def _edge_energy(edge_data: dict, time_s: float, speed_mps: float, dv_dt: float) -> float:
        # slope_angle_rad is pre-computed on each edge: positive = uphill, negative = downhill.
        alpha = float(edge_data.get("slope_angle_rad", 0.0))

        f_rolling = f * m * g * float(np.cos(alpha))
        f_slope = m * g * float(np.sin(alpha))
        f_acc = delta * m * dv_dt
        f_aero = 0.5 * cd * av * rho * (speed_mps ** 2)

        p_traction = (f_rolling + f_slope + f_acc + f_aero) * speed_mps
        return float((p_traction + p_hvac + p_aux) * time_s / 3.6e6)

    def _path_energy(path_nodes: list) -> float:
        """Sum segment energies along a path."""
        edges = list(zip(path_nodes[:-1], path_nodes[1:]))
        if not edges:
            return 0.0

        edge_data_list = [movement_graph.edges[u, v] for u, v in edges]
        base_times = [float(ed[weight_attr]) for ed in edge_data_list]
        lengths = [float(ed["length_m"]) for ed in edge_data_list]
        times = [t / s_mult for t in base_times]
        speeds = [(lengths[k] / times[k]) if times[k] > 0 else 0.0 for k in range(len(edges))]

        total = 0.0
        for k, ed in enumerate(edge_data_list):
            if k < len(edges) - 1 and times[k] > 0:
                dv_dt = (speeds[k + 1] - speeds[k]) / times[k]
            else:
                dv_dt = 0.0
            total += _edge_energy(ed, times[k], speeds[k], dv_dt)
        return total

    # ── Batch Dijkstra: one call per source → shortest path tree to ALL nodes ─
    for i, src in enumerate(service_nodes):
        # single_source_dijkstra returns (distances_dict, paths_dict) in one Dijkstra run.
        try:
            _dists, paths = nx.single_source_dijkstra(
                movement_graph, source=src, weight=weight_attr
            )
        except nx.NetworkXError:
            for j in range(n):
                energy_kwh[i, j] = 0.0 if i == j else np.inf
            continue

        for j, tgt in enumerate(service_nodes):
            if i == j:
                continue
            path = paths.get(tgt)
            if path is None:
                energy_kwh[i, j] = np.inf
            else:
                energy_kwh[i, j] = _path_energy(path)

    return energy_kwh

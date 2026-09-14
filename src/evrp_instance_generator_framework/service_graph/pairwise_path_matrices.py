from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from ..types import Period


def compute_pairwise_path_matrices(
    movement_graph: nx.DiGraph,
    service_nodes: List[int],
    periods: Tuple[Period, ...] = ("off_peak", "midday", "pm_peak"),
    *,
    include_distance_matrix: bool = True,
) -> Tuple[Dict[Period, np.ndarray], Optional[np.ndarray]]:
    """
    Compute:
    - travel-time matrices (one per period)
    - distance matrix (length_m)

    Matrices are aligned with `service_nodes` ordering.
    """

    n = len(service_nodes)
    node_to_idx = {nid: i for i, nid in enumerate(service_nodes)}

    travel_time_matrices_s: Dict[Period, np.ndarray] = {
        p: np.full((n, n), np.inf, dtype=float) for p in periods
    }
    distance_matrix_m: Optional[np.ndarray] = None
    if include_distance_matrix:
        distance_matrix_m = np.full((n, n), np.inf, dtype=float)
        # Distance matrix: one Dijkstra per service node
        for i, src in enumerate(service_nodes):
            dist_len = nx.single_source_dijkstra_path_length(
                movement_graph, source=src, weight="length_m"
            )
            distance_matrix_m[i, i] = 0.0
            for nid, dist in dist_len.items():
                j = node_to_idx.get(nid)
                if j is not None:
                    distance_matrix_m[i, j] = float(dist)

    # Travel-time matrices: one Dijkstra per (period, service_node)
    for p in periods:
        weight_attr = f"{p}_travel_time_s"
        mat = travel_time_matrices_s[p]
        for i, src in enumerate(service_nodes):
            dist_tt = nx.single_source_dijkstra_path_length(
                movement_graph, source=src, weight=weight_attr
            )
            mat[i, i] = 0.0
            for nid, tt in dist_tt.items():
                j = node_to_idx.get(nid)
                if j is not None:
                    mat[i, j] = float(tt)

    return travel_time_matrices_s, distance_matrix_m

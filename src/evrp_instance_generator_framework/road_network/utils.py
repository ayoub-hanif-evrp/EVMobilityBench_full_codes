"""
Road-graph preparation and service-matrix helpers shared across EVRP variants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from ..utils.snapping import snap_single_point
from ..types import EVFeatures, Period


def build_disk_cache(enabled: bool, cache_dir: Optional[str]):
    """Construct an ``OsmDiskCache`` or return ``None``."""
    if not enabled:
        return None
    from ..data.osm_disk_cache import OsmDiskCache, default_cache_dir

    root = Path(cache_dir).expanduser().resolve() if cache_dir else default_cache_dir()
    return OsmDiskCache(root)


def download_graph(
    city: str,
    country: str,
    disk_cache=None,
    *,
    osm_network_type: str = "drive",
    osm_retain_all: bool = True,
) -> nx.DiGraph:
    """Download a directed drive graph for *city, country* (see ``GenerationConfig`` OSM fields)."""
    from ..road_network.osm_graph_download import download_directed_drive_graph

    return download_directed_drive_graph(
        city,
        country,
        disk_cache,
        network_type=osm_network_type,
        retain_all=osm_retain_all,
    )


def prepare_movement_graph(
    G: nx.DiGraph,
    *,
    elevation_provider: str = "open_elevation",
) -> nx.DiGraph:
    """
    Full preparation pipeline shared by all variants:
    SCC → elevation → edge attributes.
    """
    from ..road_network.edge_speed_and_travel_time import assign_edge_attributes
    from ..road_network.node_elevation import attach_node_elevation
    from ..road_network.strongly_connected_component import largest_strongly_connected_component

    G = largest_strongly_connected_component(G)
    G = attach_node_elevation(G, provider=elevation_provider)
    G = assign_edge_attributes(G)
    return G


def graph_bbox(G: nx.DiGraph) -> Tuple[float, float, float, float]:
    """Bounding box ``(west, south, east, north)`` from graph node coords."""
    lats = [float(d["y"]) for _, d in G.nodes(data=True)]
    lons = [float(d["x"]) for _, d in G.nodes(data=True)]
    return min(lons), min(lats), max(lons), max(lats)


def snap_depot(
    lat: float,
    lon: float,
    movement_graph: nx.DiGraph,
    max_dist_m: float,
) -> Tuple[int, float]:
    """
    Map a depot **facility** (building site, geocode, or map pin in WGS-84) to the
    nearest **legal drivable** graph vertex within ``max_dist_m``.

    Shortest-path travel times and matrices use this node as the depot's road
    access point (vehicle enters/leaves the facility via that intersection).
    """
    return snap_single_point(lat, lon, movement_graph, max_dist_m)


def depot_single_source_times(
    movement_graph: nx.DiGraph,
    depot_node_id: int,
    weight_attr: str,
) -> Dict[int, float]:
    """Dijkstra from depot to all reachable nodes.  Returns ``{node_id: time_s}``."""
    from networkx import single_source_dijkstra_path_length

    return dict(
        single_source_dijkstra_path_length(
            movement_graph, source=depot_node_id, weight=weight_attr
        )
    )


def depot_return_times_to_depot(
    movement_graph: nx.DiGraph,
    depot_node_id: int,
    weight_attr: str,
) -> Dict[int, float]:
    """
    Shortest travel time from each node *v* to *depot_node_id* along ``G``.

    Equivalent to running Dijkstra from the depot on the edge-reversed graph.
    """
    return depot_single_source_times(
        movement_graph.reverse(copy=True), depot_node_id, weight_attr,
    )


def download_road_network(
    city: str,
    country: str,
    *,
    disk_cache=None,
    use_disk_cache: bool = True,
    osm_network_type: str = "drive",
    osm_retain_all: bool = True,
) -> nx.DiGraph:
    """
    Load a directed **drive** network for ``"{city}, {country}"`` (OSMnx).

    By default uses the standard disk cache; pass ``use_disk_cache=False`` to disable,
    or ``disk_cache=...`` for an explicit cache object.

    Does **not** apply largest-SCC filtering, elevation, or edge speed attributes —
    use :func:`prepare_movement_graph` for that.
    """
    from ..road_network.osm_graph_download import download_directed_drive_graph

    cache = disk_cache
    if cache is None and use_disk_cache:
        cache = build_disk_cache(True, None)
    elif not use_disk_cache:
        cache = None
    return download_directed_drive_graph(
        city,
        country,
        disk_cache=cache,
        network_type=osm_network_type,
        retain_all=osm_retain_all,
    )


def movement_graph_bbox(G: nx.DiGraph) -> Tuple[float, float, float, float]:
    """Alias for :func:`graph_bbox`."""
    return graph_bbox(G)


def compute_service_matrices(
    movement_graph: nx.DiGraph,
    service_nodes: List[int],
    *,
    ev_features: EVFeatures,
    energy_period: Period = "off_peak",
    periods: Tuple[Period, ...] = ("off_peak", "midday", "pm_peak"),
    include_distance_matrix: bool = True,
    include_energy: bool = True,
) -> Tuple[Optional[np.ndarray], Dict[Period, np.ndarray], Optional[np.ndarray]]:
    """
    Pairwise distance (optional), travel time (per period), and energy (optional)
    on the ordered list ``service_nodes``.

    Travel times are **not** divided by ``ev_features.speed_multiplier`` here; divide
    matrices used for time windows by ``ev_features.speed_multiplier`` yourself for parity
    with :func:`evrp_benchmark.generate_instance`.

    Energy uses the same physics as the full pipeline via
    :func:`service_graph.energy_consumption.compute_energy_matrix`.
    """
    from ..service_graph.energy_consumption import compute_energy_matrix
    from ..service_graph.pairwise_path_matrices import compute_pairwise_path_matrices

    travel_time_matrices_s, distance_m = compute_pairwise_path_matrices(
        movement_graph,
        service_nodes,
        periods=periods,
        include_distance_matrix=include_distance_matrix,
    )
    energy_kwh: Optional[np.ndarray] = None
    if include_energy:
        tt_for_energy = travel_time_matrices_s.get(energy_period)
        energy_kwh = compute_energy_matrix(
            movement_graph,
            service_nodes,
            period=energy_period,
            ev_features=ev_features,
            precomputed_travel_time=tt_for_energy,
        )
    return distance_m, travel_time_matrices_s, energy_kwh

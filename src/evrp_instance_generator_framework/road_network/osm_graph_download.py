import os
from typing import Any, Optional

import networkx as nx
import osmnx as ox

from ..data.osm_disk_cache import OsmDiskCache, save_drive_graph, try_load_drive_graph


def _graph_disk_profile(network_type: str, retain_all: bool) -> str:
    """Cache key fragment — must change when download semantics change."""
    nt = str(network_type).strip().casefold().replace(" ", "_")
    return f"osm_{nt}_retain{int(bool(retain_all))}_v4"

# Large cities (Casablanca, Istanbul, ...) exceed the default 2.5 km² Overpass
# tile limit, causing OSMnx to split into dozens of sub-queries (very slow).
# Raising the limit lets the server handle it in one pass.
ox.settings.max_query_area_size = 50 * 1_000_000_000  # 50 billion sq m ≈ no practical limit

# Optional faster / private Overpass endpoint (no Docker required).
# Example: export EVRP_OVERPASS_URL="https://overpass.kumi.systems/api/interpreter"
_overpass = os.environ.get("EVRP_OVERPASS_URL", "").strip()
if _overpass:
    ox.settings.overpass_url = _overpass


def download_directed_drive_graph(
    city: str,
    country: str,
    disk_cache: Optional[OsmDiskCache] = None,
    *,
    network_type: str = "drive",
    retain_all: bool = True,
) -> nx.DiGraph:
    """
    Download a directed drivable road network using OSMnx.

    Default ``network_type="drive"`` (public roads; fewer port/service spurs). Use
    ``"drive_service"`` to include ``highway=service`` when dense connectivity matters.
    ``retain_all`` mirrors OSMnx: when False, only the largest weakly connected fragment
    is kept at download time.

    Returns a `networkx.DiGraph` (not MultiDiGraph) with at least:
    - `length` on edges
    - OSM attributes like `highway` and optionally `maxspeed` on edges
    - `x` (lon) and `y` (lat) on nodes
    """
    profile = _graph_disk_profile(network_type, retain_all)

    cached = try_load_drive_graph(
        city, country, disk_cache, graph_profile=profile,
    )
    if cached is not None:
        G = cached
    else:
        place = f"{city}, {country}"
        G = ox.graph_from_place(
            place,
            network_type=network_type,
            simplify=True,
            retain_all=retain_all,
        )

    # OSMnx returns a MultiDiGraph. Your downstream code expects a plain DiGraph,
    # so we convert by keeping (u, v) only once.
    # If multiple parallel edges exist, we keep the one with the smallest `length`.
    if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
        H = nx.DiGraph()
        # Preserve graph-level metadata (OSMnx relies on `graph["crs"]`).
        H.graph.update(G.graph)
        H.add_nodes_from(G.nodes(data=True))

        for u, v, _, data in G.edges(keys=True, data=True):
            if not H.has_edge(u, v):
                H.add_edge(u, v, **data)
                continue

            # Choose edge with smaller length when possible.
            cur = H.edges[u, v]
            cur_len = float(cur.get("length", float("inf")))
            new_len = float(data.get("length", float("inf")))
            if new_len < cur_len:
                H.add_edge(u, v, **data)

        # Ensure CRS exists so OSMnx conversion utilities work.
        if "crs" not in H.graph:
            H.graph["crs"] = "EPSG:4326"

        G = H

    # Ensure latitude/longitude exist.
    for _, data in G.nodes(data=True):
        if "x" not in data or "y" not in data:
            raise ValueError("OSMnx graph nodes are missing 'x'/'y' coordinates.")

    if cached is None:
        save_drive_graph(
            city, country, G, disk_cache, graph_profile=profile,
        )

    return G


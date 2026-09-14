from typing import Tuple

import networkx as nx
import geopandas as gpd


def movement_graph_to_nodes_gdf(G: nx.DiGraph) -> gpd.GeoDataFrame:
    """
    Convert movement graph nodes to a GeoDataFrame.
    """

    rows = []
    for n, data in G.nodes(data=True):
        # OSMnx uses x=lon, y=lat
        rows.append({"node_id": n, **data})
    gdf = gpd.GeoDataFrame(rows, geometry=gpd.points_from_xy([r["x"] for r in rows], [r["y"] for r in rows]), crs="EPSG:4326")
    return gdf


def movement_graph_to_edges_gdf(G: nx.DiGraph) -> gpd.GeoDataFrame:
    """
    Convert movement graph edges to a GeoDataFrame using straight LineStrings.

    Note: For a minimal implementation, we do not recover OSM polylines.
    """

    rows = []
    for u, v, data in G.edges(data=True):
        rows.append({"u": u, "v": v, **data})

    # Build geometry from OSMnx edge geometry when available.
    # OSMnx often stores a shapely LineString in `edge_data["geometry"]`.
    from shapely.geometry import LineString

    geoms = []
    for r in rows:
        u = r["u"]
        v = r["v"]
        edge_data = G.edges[u, v]
        geom = edge_data.get("geometry")
        if geom is not None and hasattr(geom, "coords"):
            coords = list(geom.coords)
            geoms.append(LineString([(float(x), float(y)) for (x, y) in coords]))
        else:
            u_data = G.nodes[u]
            v_data = G.nodes[v]
            geoms.append(LineString([(u_data["x"], u_data["y"]), (v_data["x"], v_data["y"])]))

    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
    return gdf


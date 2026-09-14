"""
Maps and feasibility text for scripts and Jupyter.

**Simple pattern:** ``plot_...(graph, to_file=None)`` — leave ``to_file`` out (or ``None``) to show the
figure in a notebook; pass a path string to save a PNG instead.

Requires matplotlib (comes with geopandas).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple, Union

import networkx as nx

from .export.graph_to_geodataframes import movement_graph_to_edges_gdf, movement_graph_to_nodes_gdf
from .road_network.utils import download_road_network, movement_graph_bbox, prepare_movement_graph
from .types import depot_facility_latlon, primary_depot_facility_latlon

PathLike = Union[str, Path]

RoadView = Literal["full", "edges", "nodes"]
RoadsBehind = Literal["edges", "full", "hide"]
RoadPalette = Literal["web", "dual", "unified"]

# Folium overlay sizing (screen pixels).
_INTERACTIVE_ROAD_BLUE = "#4278df"
_INTERACTIVE_EDGE_WEIGHT = 1.8
# CircleMarker radius is in px at the current zoom — keep junction dots tiny (full-city view).
_ROAD_NODE_RADIUS_DUAL = 1
_ROAD_NODE_RADIUS_UNIFIED = 2
# Previously used for custom ``iconSize`` — removed: AwesomeMarkers defaults must stay in sync
# with ``iconAnchor`` / mask size; wrong dimensions clip pins to half-moons in the notebook.
# Leaflet.AwesomeMarkers.Icon: https://github.com/lvoogdt/Leaflet.awesome-markers

# ---------------------------------------------------------------------------
# OSM road styling — same tables as ``web/Streamlit_app/app.py`` PyDeck maps
# (Step 3 preview + Step 5 results: PathLayer + ScatterplotLayer junctions).
# ---------------------------------------------------------------------------
_HIGHWAY_RANK = (
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "track",
    "pedestrian",
)

_HIGHWAY_STYLE = {
    "motorway": ([220, 55, 55], 3.2),
    "motorway_link": ([235, 95, 95], 2.6),
    "trunk": ([215, 110, 45], 2.8),
    "trunk_link": ([235, 145, 75], 2.3),
    "primary": ([255, 165, 40], 2.4),
    "primary_link": ([255, 200, 110], 2.0),
    "secondary": ([245, 210, 95], 2.0),
    "secondary_link": ([250, 225, 140], 1.7),
    "tertiary": ([200, 195, 130], 1.6),
    "tertiary_link": ([215, 210, 160], 1.5),
    "unclassified": ([155, 165, 155], 1.5),
    "residential": ([130, 155, 205], 1.4),
    "living_street": ([120, 175, 195], 1.3),
    "service": ([145, 145, 168], 1.2),
    "track": ([110, 135, 115], 1.2),
    "pedestrian": ([175, 175, 195], 1.2),
}


def _highway_key_web(data: Mapping[str, Any]) -> str:
    hw = data.get("highway")
    if hw is None:
        return "unclassified"
    if isinstance(hw, (list, tuple, set)):
        tags = [str(x) for x in hw]
        for rank in _HIGHWAY_RANK:
            if rank in tags:
                return rank
        return str(tags[0]) if tags else "unclassified"
    return str(hw)


def _highway_color_width_rgb(data: Mapping[str, Any]) -> Tuple[list[int], float]:
    key = _highway_key_web(data)
    pair = _HIGHWAY_STYLE.get(key)
    if pair is not None:
        return pair[0], float(pair[1])
    for pref in _HIGHWAY_RANK:
        if key.startswith(pref):
            fb = _HIGHWAY_STYLE.get(pref, ([155, 155, 155], 1.4))
            return fb[0], float(fb[1])
    return [125, 120, 115], 1.4


def _rgb_seq_to_hex(rgb: Sequence[int]) -> str:
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return f"#{r:02x}{g:02x}{b:02x}"


def _folium_line_weight_from_pydeck(deck_width: float) -> float:
    """Deck uses ~1.2–3.2; Folium ``weight`` is coarse (px)."""
    return float(max(1, min(5, round(deck_width))))


# Step 5 ``ScatterplotLayer`` on graph nodes in the Streamlit results map (PyDeck px scale).
# Folium uses screen pixels too: keep **1** so city-scale views stay readable; zoom in to see detail.
_WEB_JUNCTION_DOT_RGB: Tuple[int, int, int] = (190, 215, 255)
_WEB_JUNCTION_RADIUS_PX = 1
# Light overlay: dense graphs otherwise look like a solid blob and blow up saved notebook HTML.
_WEB_JUNCTION_FILL_OPACITY_FOLIUM = 0.38

# Matches ``_DEPOT_MAP_COLOR = [255, 30, 30]`` in the app (Font Awesome home icon, red pin).
_FOLIUM_DEPOT_ICON_COLOR = "darkred"

# Bootstrap 3 glyphicons (Folium’s usual default) — no FA4/FA5 mismatch.
_FOLIUM_GLYPH_DEPOT = "home"
_FOLIUM_GLYPH_CUSTOMER = "user"
_FOLIUM_GLYPH_STATION = "off"  # power (charging)
_FOLIUM_GLYPH_SATELLITE = "inbox"  # depot / secondary facility


def _highway_key_edge_counts(graph: nx.DiGraph) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, _, d in graph.edges(data=True):
        k = _highway_key_web(d)
        counts[k] = counts.get(k, 0) + 1
    return counts


def _order_highway_keys(keys: set[str]) -> list[str]:
    ordered: list[str] = []
    keys = set(keys)
    for hw in _HIGHWAY_RANK:
        if hw in keys:
            ordered.append(hw)
            keys.discard(hw)
    for hw in sorted(keys):
        ordered.append(hw)
    return ordered


def _highway_keys_for_legend(graph: nx.DiGraph) -> list[str]:
    """Highway classes frequent enough to list (same idea as the Streamlit legend)."""
    counts = _highway_key_edge_counts(graph)
    if not counts:
        return []
    n_e = sum(counts.values())
    floor = max(30, min(400, int(0.0015 * max(n_e, 1))))
    frequent = {k for k, c in counts.items() if c >= floor}
    if not frequent:
        top = sorted(counts.items(), key=lambda x: -x[1])[:8]
        frequent = {k for k, _ in top}
    return _order_highway_keys(frequent)


def _highway_legend_label(hw: str) -> str:
    return hw.replace("_", " ").title()


def prepare_city_road_network(
    city: str,
    country: str,
    *,
    flat_terrain: bool = True,
    use_disk_cache: bool = True,
) -> nx.DiGraph:
    """
    **Step 1 (web app):** load the drivable road network for a city.

    - ``flat_terrain=True`` — fast, no hill slopes in energy (recommended for first tries).
    - ``flat_terrain=False`` — real elevations from Open-Elevation (slower, needs internet).

    The OSM graph is cached on disk automatically.
    """
    elev: Literal["none", "open_elevation"] = "none" if flat_terrain else "open_elevation"
    raw = download_road_network(city, country, use_disk_cache=use_disk_cache)
    return prepare_movement_graph(raw, elevation_provider=elev)


def geographic_center_of_graph(graph: nx.DiGraph) -> Tuple[float, float]:
    """Middle of the bounding box: ``(latitude, longitude)`` — handy as a default depot point."""
    west, south, east, north = movement_graph_bbox(graph)
    return (south + north) / 2.0, (west + east) / 2.0


def _require_folium():
    try:
        import folium
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "Interactive map functions require folium. Install with: pip install folium"
        ) from exc
    return folium


def _folium_fit_to_graph(fmap: Any, graph: nx.DiGraph) -> None:
    """Zoom the map to the graph extent so pan/zoom starts in the right area."""
    west, south, east, north = movement_graph_bbox(graph)
    lon_span = max(east - west, 1e-6)
    lat_span = max(north - south, 1e-6)
    pad_lon = max(lon_span * 0.08, 0.001)
    pad_lat = max(lat_span * 0.08, 0.001)
    fmap.fit_bounds(
        [[south - pad_lat, west - pad_lon], [north + pad_lat, east + pad_lon]],
        padding=(12, 12),
        max_zoom=18,
    )


def _folium_notebook_interactivity(fmap: Any) -> None:
    """Fullscreen + zoom controls so the map behaves clearly as an interactive Leaflet widget."""
    try:
        from folium.plugins import Fullscreen

        Fullscreen(position="topright", title="Fullscreen", title_cancel="Exit").add_to(fmap)
    except Exception:  # noqa: BLE001
        pass


def _folium_marker_icon(
    folium_mod: Any,
    *,
    color: str,
    icon: str,
    prefix: str = "glyphicon",
) -> Any:
    """Colored teardrop markers (Bootstrap Glyphicons by default).

    Do **not** pass ``iconSize`` unless you also set matching ``iconAnchor`` / ``shadow`` options;
    mis-sized icons clip to half-circles in Leaflet.AwesomeMarkers.
    """
    return folium_mod.Icon(
        color=color,
        icon_color="white",
        icon=icon,
        prefix=prefix,
    )


def _edge_latlon_path(graph: nx.DiGraph, u: int, v: int, data: Mapping[str, Any]) -> list[list[float]]:
    xu, yu = float(graph.nodes[u]["x"]), float(graph.nodes[u]["y"])
    xv, yv = float(graph.nodes[v]["x"]), float(graph.nodes[v]["y"])
    geom = data.get("geometry")
    if geom is not None and hasattr(geom, "coords"):
        path = [[float(y), float(x)] for (x, y) in geom.coords]
        if path:
            path[0] = [yu, xu]
            path[-1] = [yv, xv]
            return path
    return [[yu, xu], [yv, xv]]


def _is_major_road(data: Mapping[str, Any]) -> bool:
    hw = data.get("highway")
    if isinstance(hw, (list, tuple, set)):
        tags = {str(x) for x in hw}
    else:
        tags = {str(hw)} if hw is not None else set()
    major = {
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
    }
    return any(t in major for t in tags)


def _add_interactive_legend(
    fmap: Any,
    *,
    road_palette: RoadPalette,
    graph: Optional[nx.DiGraph],
    show_nodes: bool,
    show_customers: bool,
    show_stations: bool,
    show_satellites: bool,
    show_depot: bool,
) -> None:
    legend_rows: list[tuple[str, str]] = []
    if road_palette == "web" and graph is not None:
        for hw in _highway_keys_for_legend(graph):
            rgb, _w = _highway_color_width_rgb({"highway": hw})
            legend_rows.append((_rgb_seq_to_hex(rgb), _highway_legend_label(hw)))
        if show_nodes:
            legend_rows.append((_rgb_seq_to_hex(list(_WEB_JUNCTION_DOT_RGB)), "Road junctions"))
    elif road_palette == "unified":
        legend_rows = [(_INTERACTIVE_ROAD_BLUE, "Road network")]
        if show_nodes:
            legend_rows.append((_INTERACTIVE_ROAD_BLUE, "Road junctions"))
    else:
        legend_rows = [
            ("#f59e0b", "Major roads"),
            ("#4a7fd1", "Local roads"),
        ]
        if show_nodes:
            legend_rows.append(("#b7caf1", "Road junctions"))

    depot_swatch = "#ff1e1e" if road_palette == "web" else "#1d4ed8"
    if show_depot:
        legend_rows.append((depot_swatch, "Depot"))
    if show_customers:
        legend_rows.append(("#ea580c", "Customers"))
    if show_stations:
        legend_rows.append(("#16a34a", "Stations"))
    if show_satellites:
        legend_rows.append(("#7e22ce", "Satellites"))

    rows_html = "".join(
        f'<div style="margin:2px 0;"><span style="display:inline-block;width:10px;height:10px;'
        f'background:{color};border:1px solid #666;margin-right:6px;"></span>{label}</div>'
        for color, label in legend_rows
    )
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 24px;
        left: 24px;
        z-index: 9999;
        background: rgba(255,255,255,0.95);
        border: 1px solid #ccc;
        border-radius: 6px;
        padding: 10px 12px;
        font-size: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15);
    ">
      <div style="font-weight:700;margin-bottom:6px;">Map Legend</div>
      {rows_html}
    </div>
    """
    folium = _require_folium()
    fmap.get_root().html.add_child(folium.Element(legend_html))


def map_city_roads_interactive(
    graph: nx.DiGraph,
    *,
    max_edges: Optional[int] = None,
    max_nodes: Optional[int] = 8_000,
    show_nodes: bool = True,
    road_palette: RoadPalette = "web",
    tiles: Optional[str] = None,
    zoom_start: int = 13,
    show_legend: bool = True,
    unified_road_style: Optional[bool] = None,
    map_height: str = "560px",
) -> Any:
    """Interactive road map for Jupyter (Leaflet pan/zoom).

    Default ``road_palette=\"web\"`` matches Step 5 in the Streamlit app: OSM highway colors from
    ``_HIGHWAY_STYLE``, light Carto-style basemap unless ``tiles`` is set, tiny light junction dots
    (see ``_WEB_JUNCTION_*``). Use ``road_palette=\"dual\"`` or ``\"unified\"`` for older styles.
    ``max_nodes`` caps how many junction markers are drawn (smaller = smaller saved notebook size);
    pass ``max_nodes=None`` only if you need every node (outputs can be very large).
    If ``unified_road_style`` is set (deprecated), it maps to ``\"unified\"`` / ``\"dual\"``.
    """
    rp: RoadPalette = road_palette
    if unified_road_style is not None:
        rp = "unified" if unified_road_style else "dual"

    folium = _require_folium()
    c_lat, c_lon = geographic_center_of_graph(graph)
    effective_tiles = (
        tiles if tiles is not None else ("CartoDB positron" if rp == "web" else "OpenStreetMap")
    )
    fmap = folium.Map(
        location=[c_lat, c_lon],
        zoom_start=zoom_start,
        tiles=effective_tiles,
        control_scale=True,
        height=map_height,
        prefer_canvas=False,
    )
    _folium_notebook_interactivity(fmap)

    edges = list(graph.edges(data=True))
    if max_edges is not None and len(edges) > max_edges:
        import random

        random.seed(0)
        edges = random.sample(edges, k=max_edges)

    for u, v, d in edges:
        if rp == "web":
            rgb, deck_w = _highway_color_width_rgb(d)
            edge_color = _rgb_seq_to_hex(rgb)
            weight = _folium_line_weight_from_pydeck(deck_w)
            opacity = 0.92
        elif rp == "unified":
            edge_color = _INTERACTIVE_ROAD_BLUE
            weight = float(_INTERACTIVE_EDGE_WEIGHT)
            opacity = 0.92
        else:
            is_major = _is_major_road(d)
            edge_color = "#f59e0b" if is_major else "#4a7fd1"
            weight = 2.0 if is_major else 1.0
            opacity = 0.9 if is_major else 0.75
        folium.PolyLine(
            locations=_edge_latlon_path(graph, u, v, d),
            color=edge_color,
            weight=weight,
            opacity=opacity,
        ).add_to(fmap)
    if show_nodes:
        nodes = list(graph.nodes(data=True))
        if max_nodes is not None and len(nodes) > max_nodes:
            import random

            random.seed(1)
            nodes = random.sample(nodes, k=max_nodes)
        if rp == "web":
            node_fill = _rgb_seq_to_hex(list(_WEB_JUNCTION_DOT_RGB))
            node_r = float(_WEB_JUNCTION_RADIUS_PX)
            fill_opacity = float(_WEB_JUNCTION_FILL_OPACITY_FOLIUM)
            outline_w = 0
        elif rp == "unified":
            node_fill = _INTERACTIVE_ROAD_BLUE
            node_r = float(_ROAD_NODE_RADIUS_UNIFIED)
            fill_opacity = 0.55
            outline_w = 1
        else:
            node_fill = "#b7caf1"
            node_r = float(_ROAD_NODE_RADIUS_DUAL)
            fill_opacity = 0.45
            outline_w = 0
        for _, ndata in nodes:
            folium.CircleMarker(
                location=[float(ndata["y"]), float(ndata["x"])],
                radius=node_r,
                color=node_fill,
                fill=True,
                fill_color=node_fill,
                fill_opacity=fill_opacity,
                weight=outline_w,
            ).add_to(fmap)

    try:
        _folium_fit_to_graph(fmap, graph)
    except Exception:  # noqa: BLE001
        pass

    if show_legend:
        _add_interactive_legend(
            fmap,
            road_palette=rp,
            graph=graph if rp == "web" else None,
            show_nodes=show_nodes,
            show_customers=False,
            show_stations=False,
            show_satellites=False,
            show_depot=False,
        )
    return fmap


def map_city_roads_with_depot_interactive(
    graph: nx.DiGraph,
    depot_latitude: float,
    depot_longitude: float,
    *,
    max_edges: Optional[int] = None,
    max_nodes: Optional[int] = 8_000,
    show_nodes: bool = True,
    tiles: Optional[str] = None,
    zoom_start: int = 13,
) -> Any:
    """Interactive road map + depot marker (web-style roads + red depot pin, same as Streamlit Step 5)."""
    folium = _require_folium()
    fmap = map_city_roads_interactive(
        graph,
        max_edges=max_edges,
        max_nodes=max_nodes,
        show_nodes=show_nodes,
        road_palette="web",
        tiles=tiles,
        zoom_start=zoom_start,
        show_legend=False,
    )
    folium.Marker(
        [float(depot_latitude), float(depot_longitude)],
        tooltip="Depot (facility)",
        icon=_folium_marker_icon(folium, color=_FOLIUM_DEPOT_ICON_COLOR, icon=_FOLIUM_GLYPH_DEPOT),
    ).add_to(fmap)
    _add_interactive_legend(
        fmap,
        road_palette="web",
        graph=graph,
        show_nodes=show_nodes,
        show_customers=False,
        show_stations=False,
        show_satellites=False,
        show_depot=True,
    )
    return fmap


def map_services_interactive(
    graph: nx.DiGraph,
    depot_latitude: float,
    depot_longitude: float,
    *,
    customers: Optional[Sequence[Any]] = None,
    stations: Optional[Sequence[Any]] = None,
    max_edges: Optional[int] = None,
    max_nodes: Optional[int] = 8_000,
    show_nodes: bool = True,
    tiles: Optional[str] = None,
    zoom_start: int = 13,
) -> Any:
    """Interactive map for depot + customers + stations on road network."""
    folium = _require_folium()
    fmap = map_city_roads_interactive(
        graph,
        max_edges=max_edges,
        max_nodes=max_nodes,
        show_nodes=show_nodes,
        road_palette="web",
        tiles=tiles,
        zoom_start=zoom_start,
        show_legend=False,
    )
    folium.Marker(
        [float(depot_latitude), float(depot_longitude)],
        tooltip="Depot (facility)",
        icon=_folium_marker_icon(folium, color=_FOLIUM_DEPOT_ICON_COLOR, icon=_FOLIUM_GLYPH_DEPOT),
    ).add_to(fmap)

    for c in list(customers or []):
        folium.Marker(
            [float(c.lat), float(c.lon)],
            tooltip=f"Customer {getattr(c, 'id', '')}",
            icon=_folium_marker_icon(folium, color="orange", icon=_FOLIUM_GLYPH_CUSTOMER),
        ).add_to(fmap)

    for s in list(stations or []):
        folium.Marker(
            [float(s.lat), float(s.lon)],
            tooltip=f"Station {getattr(s, 'id', '')}",
            icon=_folium_marker_icon(folium, color="green", icon=_FOLIUM_GLYPH_STATION),
        ).add_to(fmap)
    _add_interactive_legend(
        fmap,
        road_palette="web",
        graph=graph,
        show_nodes=show_nodes,
        show_customers=bool(customers),
        show_stations=bool(stations),
        show_satellites=False,
        show_depot=True,
    )
    return fmap


def map_benchmark_interactive(
    instance: Any,
    *,
    max_edges: Optional[int] = None,
    max_nodes: Optional[int] = 8_000,
    show_nodes: bool = True,
    tiles: Optional[str] = None,
    zoom_start: int = 13,
) -> Any:
    """Interactive benchmark map with styled roads and service markers."""
    folium = _require_folium()
    g = instance.movement_graph
    fmap = map_city_roads_interactive(
        g,
        max_edges=max_edges,
        max_nodes=max_nodes,
        show_nodes=show_nodes,
        road_palette="web",
        tiles=tiles,
        zoom_start=zoom_start,
        show_legend=False,
    )

    depots = getattr(instance, "depots", None) or []
    if depots:
        for dep in depots:
            dlat, dlon = depot_facility_latlon(dep)
            folium.Marker(
                [float(dlat), float(dlon)],
                tooltip=f"Depot {getattr(dep, 'id', '')}",
                icon=_folium_marker_icon(folium, color=_FOLIUM_DEPOT_ICON_COLOR, icon=_FOLIUM_GLYPH_DEPOT),
            ).add_to(fmap)
    else:
        dlat, dlon = primary_depot_facility_latlon(instance)
        folium.Marker(
            [float(dlat), float(dlon)],
            tooltip="Depot (facility)",
            icon=_folium_marker_icon(folium, color=_FOLIUM_DEPOT_ICON_COLOR, icon=_FOLIUM_GLYPH_DEPOT),
        ).add_to(fmap)

    for c in list(getattr(instance, "customers", []) or []):
        folium.Marker(
            [float(c.lat), float(c.lon)],
            tooltip=f"Customer {getattr(c, 'id', '')}",
            icon=_folium_marker_icon(folium, color="orange", icon=_FOLIUM_GLYPH_CUSTOMER),
        ).add_to(fmap)

    for s in list(getattr(instance, "stations", []) or []):
        folium.Marker(
            [float(s.lat), float(s.lon)],
            tooltip=f"Station {getattr(s, 'id', '')}",
            icon=_folium_marker_icon(folium, color="green", icon=_FOLIUM_GLYPH_STATION),
        ).add_to(fmap)

    for sat in list(getattr(instance, "satellites", []) or []):
        folium.Marker(
            [float(sat.lat), float(sat.lon)],
            tooltip=f"Satellite {getattr(sat, 'id', '')}",
            icon=_folium_marker_icon(folium, color="purple", icon=_FOLIUM_GLYPH_SATELLITE),
        ).add_to(fmap)
    _add_interactive_legend(
        fmap,
        road_palette="web",
        graph=g,
        show_nodes=show_nodes,
        show_customers=bool(getattr(instance, "customers", [])),
        show_stations=bool(getattr(instance, "stations", [])),
        show_satellites=bool(getattr(instance, "satellites", [])),
        show_depot=True,
    )
    return fmap


def _plot_finish(fig, to_file: Optional[PathLike], dpi: int) -> Optional[Path]:
    import matplotlib.pyplot as plt

    if to_file is None:
        # In notebooks/frontends, closing immediately after show can suppress rendering.
        # Keep the figure alive and display explicitly when possible.
        try:
            from IPython.display import display

            display(fig)
        except Exception:  # noqa: BLE001
            plt.show()
        return None
    path = Path(to_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path.resolve()


def plot_city_roads(
    graph: nx.DiGraph,
    *,
    view: RoadView = "full",
    to_file: Optional[PathLike] = None,
    title: Optional[str] = None,
    max_edges: int = 12_000,
    max_nodes: int = 6_000,
    figsize: Tuple[float, float] = (9.0, 9.0),
    dpi: int = 130,
) -> Optional[Path]:
    """
    Draw the road network.

    - ``view="full"`` — edges + intersection dots
    - ``view="edges"`` — lines only
    - ``view="nodes"`` — dots only

    ``to_file="map.png"`` saves; ``to_file=None`` shows (Jupyter / interactive).
    """
    import matplotlib.pyplot as plt

    show_edges = view in ("full", "edges")
    show_nodes = view in ("full", "nodes")

    fig, ax = plt.subplots(figsize=figsize)
    if show_edges:
        eg = movement_graph_to_edges_gdf(graph)
        if len(eg) > max_edges:
            eg = eg.sample(n=max_edges, random_state=0)
        eg.plot(ax=ax, color="#3d3020", linewidth=0.2, alpha=0.9)
    if show_nodes:
        ng = movement_graph_to_nodes_gdf(graph)
        if len(ng) > max_nodes:
            ng = ng.sample(n=max_nodes, random_state=1)
        ng.plot(ax=ax, color="#1e5aa8", markersize=0.85, alpha=0.55)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return _plot_finish(fig, to_file, dpi)


def plot_city_roads_with_depot(
    graph: nx.DiGraph,
    depot_latitude: float,
    depot_longitude: float,
    *,
    to_file: Optional[PathLike] = None,
    show_roads: bool = True,
    title: Optional[str] = None,
    max_edges: int = 12_000,
    figsize: Tuple[float, float] = (9.0, 9.0),
    dpi: int = 130,
) -> Optional[Path]:
    """
    **Step 2 (web app):** roads plus a red depot **facility** marker at WGS84 ``(depot_latitude, depot_longitude)``
    (building / site — not necessarily the snapped graph node).

    ``show_roads=False`` draws only the depot (rarely needed).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    if show_roads:
        eg = movement_graph_to_edges_gdf(graph)
        if len(eg) > max_edges:
            eg = eg.sample(n=max_edges, random_state=0)
        eg.plot(ax=ax, color="#3d3020", linewidth=0.2, alpha=0.9)
    ax.scatter(
        depot_longitude,
        depot_latitude,
        c="#d92d20",
        s=220,
        zorder=6,
        edgecolors="white",
        linewidths=1.0,
        label="Depot (facility)",
    )
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return _plot_finish(fig, to_file, dpi)


def plot_services_on_map(
    graph: nx.DiGraph,
    depot_latitude: float,
    depot_longitude: float,
    *,
    customers: Optional[Sequence[Any]] = None,
    stations: Optional[Sequence[Any]] = None,
    to_file: Optional[PathLike] = None,
    roads: RoadsBehind = "edges",
    show_depot: bool = True,
    show_customers: bool = True,
    show_stations: bool = True,
    title: Optional[str] = None,
    max_edges: int = 12_000,
    max_nodes: int = 6_000,
    figsize: Tuple[float, float] = (9.0, 9.0),
    dpi: int = 130,
) -> Optional[Path]:
    """
    Map depot **facility** (red), optional customers (orange) and stations (green) on ``graph``.

    Use between phased generation steps when you do not have a ``BenchmarkInstance`` yet.
    Pass building / site WGS84 for ``depot_latitude`` / ``depot_longitude``.
    Pass only the layers you want; empty lists omit markers.
    """
    import matplotlib.pyplot as plt

    cust = list(customers) if customers is not None else []
    stat = list(stations) if stations is not None else []

    fig, ax = plt.subplots(figsize=figsize)

    if roads in ("edges", "full"):
        eg = movement_graph_to_edges_gdf(graph)
        if len(eg) > max_edges:
            eg = eg.sample(n=max_edges, random_state=0)
        eg.plot(ax=ax, color="#3d3020", linewidth=0.2, alpha=0.88)
    if roads == "full":
        ng = movement_graph_to_nodes_gdf(graph)
        if len(ng) > max_nodes:
            ng = ng.sample(n=max_nodes, random_state=1)
        ng.plot(ax=ax, color="#94a3b8", markersize=0.5, alpha=0.4)

    if show_depot:
        ax.scatter(
            depot_longitude,
            depot_latitude,
            c="#d92d20",
            s=200,
            zorder=6,
            edgecolors="white",
            linewidths=0.8,
            label="Depot (facility)",
        )
    if show_customers:
        for i, c in enumerate(cust):
            ax.scatter(
                c.lon,
                c.lat,
                c="#e85d04",
                s=55,
                zorder=5,
                edgecolors="white",
                linewidths=0.4,
                label="Customers" if i == 0 else None,
            )
    if show_stations:
        for i, s in enumerate(stat):
            ax.scatter(
                s.lon,
                s.lat,
                c="#15803d",
                s=75,
                zorder=5,
                edgecolors="white",
                linewidths=0.4,
                label="Stations" if i == 0 else None,
            )

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=9)

    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return _plot_finish(fig, to_file, dpi)


def plot_benchmark_on_map(
    instance: Any,
    *,
    to_file: Optional[PathLike] = None,
    roads: RoadsBehind = "edges",
    show_depot: bool = True,
    show_customers: bool = True,
    show_stations: bool = True,
    title: Optional[str] = None,
    max_edges: int = 12_000,
    max_nodes: int = 6_000,
    figsize: Tuple[float, float] = (9.0, 9.0),
    dpi: int = 130,
) -> Optional[Path]:
    """
    **Steps 5–6 (web app):** one map with depot **facility** (red; building / config pin),
    customers (orange), stations (green). Routing still uses ``depot_node_id`` on the graph.

    ``roads``:

    - ``"edges"`` — road lines only (default, less clutter)
    - ``"full"`` — lines + small road nodes
    - ``"hide"`` — only depot / customers / stations
    """
    import matplotlib.pyplot as plt

    G = instance.movement_graph
    fig, ax = plt.subplots(figsize=figsize)

    if roads in ("edges", "full"):
        eg = movement_graph_to_edges_gdf(G)
        if len(eg) > max_edges:
            eg = eg.sample(n=max_edges, random_state=0)
        eg.plot(ax=ax, color="#3d3020", linewidth=0.2, alpha=0.88)
    if roads == "full":
        ng = movement_graph_to_nodes_gdf(G)
        if len(ng) > max_nodes:
            ng = ng.sample(n=max_nodes, random_state=1)
        ng.plot(ax=ax, color="#94a3b8", markersize=0.5, alpha=0.4)

    if show_depot:
        depots = getattr(instance, "depots", None) or []
        if len(depots) > 0:
            for i, dep in enumerate(depots):
                dlat, dlon = depot_facility_latlon(dep)
                ax.scatter(
                    float(dlon),
                    float(dlat),
                    c="#d92d20",
                    s=200,
                    zorder=6,
                    edgecolors="white",
                    linewidths=0.8,
                    label="Depots (facility)" if i == 0 else None,
                )
        else:
            dlat, dlon = primary_depot_facility_latlon(instance)
            ax.scatter(
                float(dlon),
                float(dlat),
                c="#d92d20",
                s=200,
                zorder=6,
                edgecolors="white",
                linewidths=0.8,
                label="Depot (facility)",
            )
    if show_customers:
        for i, c in enumerate(instance.customers):
            ax.scatter(
                c.lon,
                c.lat,
                c="#e85d04",
                s=55,
                zorder=5,
                edgecolors="white",
                linewidths=0.4,
                label="Customers" if i == 0 else None,
            )
    if show_stations:
        for i, s in enumerate(instance.stations):
            ax.scatter(
                s.lon,
                s.lat,
                c="#15803d",
                s=75,
                zorder=5,
                edgecolors="white",
                linewidths=0.4,
                label="Stations" if i == 0 else None,
            )

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title or f"{instance.config.city}, {instance.config.country} — benchmark map")
    fig.tight_layout()
    return _plot_finish(fig, to_file, dpi)


def display_feasibility_summary(feasibility: Mapping[str, Any]) -> None:
    """Print three-tier feasibility (validity / time / energy) in plain language."""
    print("\n── Feasibility ──")
    if "feasibility_report_mode" in feasibility:
        print(f"  mode: {feasibility['feasibility_report_mode']}")
    if "all_passed" in feasibility:
        print(f"  all_passed: {feasibility['all_passed']}")
    for block in ("validity", "time_feasibility", "energy_feasibility"):
        if block not in feasibility:
            continue
        sub = feasibility[block]
        if not isinstance(sub, Mapping):
            continue
        ok = sub.get("ok")
        skipped = sub.get("skipped")
        label = f"  {block}"
        if skipped:
            label += " (skipped)"
        elif ok is not None:
            label += f" ok={ok}"
        print(label)
        for key in ("reason", "mode", "battery_capacity_kwh", "num_customers", "num_depots"):
            if key in sub:
                print(f"    {key}: {sub[key]}")
        issues = sub.get("issues") or []
        if issues:
            print("    issues:")
            for i, item in enumerate(issues, 1):
                print(f"      {i}. {item}")
    if "satellite_reachability" in feasibility:
        sr = feasibility["satellite_reachability"]
        if isinstance(sr, Mapping):
            print(f"  satellite_reachability ok={sr.get('ok')} skipped={sr.get('skipped')}")
            for item in sr.get("issues") or []:
                print(f"    - {item}")
    for key in ("note", "depot_node_id", "depot_count", "satellite_count"):
        if key in feasibility:
            print(f"  {key}: {feasibility[key]}")
    print("── end ──\n")


# ── Backward-compatible names (older scripts / docs) ─────────────────────────


def load_prepared_graph(
    city: str,
    country: str,
    *,
    node_elevation_provider: Literal["none", "open_elevation"] = "none",
    use_disk_cache: bool = True,
) -> nx.DiGraph:
    """Same as :func:`prepare_city_road_network` but with explicit elevation provider name."""
    return prepare_city_road_network(
        city,
        country,
        flat_terrain=(node_elevation_provider == "none"),
        use_disk_cache=use_disk_cache,
    )


graph_center_latlon = geographic_center_of_graph


def save_road_network_figure(
    graph: nx.DiGraph,
    path: PathLike,
    *,
    show_edges: bool = True,
    show_nodes: bool = True,
    title: Optional[str] = None,
    **kwargs: Any,
) -> Path:
    if show_edges and show_nodes:
        view: RoadView = "full"
    elif show_edges:
        view = "edges"
    else:
        view = "nodes"
    out = plot_city_roads(graph, view=view, to_file=path, title=title, **kwargs)
    assert out is not None
    return out


def save_road_network_with_markers(
    graph: nx.DiGraph,
    path: PathLike,
    markers: Any,
    *,
    show_edges: bool = True,
    show_nodes: bool = False,
    title: Optional[str] = None,
    **kwargs: Any,
) -> Path:
    import matplotlib.pyplot as plt

    figsize = kwargs.pop("figsize", (9.0, 9.0))
    max_edges = kwargs.pop("max_edges", 12_000)
    max_nodes = kwargs.pop("max_nodes", 6_000)
    dpi = kwargs.pop("dpi", 130)
    fig, ax = plt.subplots(figsize=figsize)
    if show_edges:
        eg = movement_graph_to_edges_gdf(graph)
        if len(eg) > max_edges:
            eg = eg.sample(n=max_edges, random_state=0)
        eg.plot(ax=ax, color="#3d3020", linewidth=0.2, alpha=0.88)
    if show_nodes:
        ng = movement_graph_to_nodes_gdf(graph)
        if len(ng) > max_nodes:
            ng = ng.sample(n=max_nodes, random_state=1)
        ng.plot(ax=ax, color="#94a3b8", markersize=0.6, alpha=0.45)
    for m in markers:
        ax.scatter(
            float(m["lon"]),
            float(m["lat"]),
            c=m.get("color", "red"),
            s=float(m.get("size", 45)),
            zorder=5,
            edgecolors="white",
            linewidths=0.6,
            label=m.get("label"),
        )
    h, lab = ax.get_legend_handles_labels()
    if lab:
        by_label = dict(zip(lab, h))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path.resolve()


def save_instance_overview_map(instance: Any, path: PathLike, **kwargs: Any) -> Path:
    """Legacy wrapper: prefer :func:`plot_benchmark_on_map` with ``to_file=...``."""
    roads = kwargs.pop("show_edges", True)
    show_road_nodes = kwargs.pop("show_road_nodes", False)
    r: RoadsBehind
    if not roads:
        r = "hide"
    elif show_road_nodes:
        r = "full"
    else:
        r = "edges"
    out = plot_benchmark_on_map(
        instance,
        to_file=path,
        roads=r,
        show_depot=kwargs.pop("show_depot", True),
        show_customers=kwargs.pop("show_customers", True),
        show_stations=kwargs.pop("show_stations", True),
        title=kwargs.pop("title", None),
        **kwargs,
    )
    assert out is not None
    return out


def print_feasibility_report(feasibility: Mapping[str, Any]) -> None:
    """Alias for :func:`display_feasibility_summary`."""
    display_feasibility_summary(feasibility)


__all__ = [
    "display_feasibility_summary",
    "geographic_center_of_graph",
    "graph_center_latlon",
    "load_prepared_graph",
    "map_benchmark_interactive",
    "map_city_roads_interactive",
    "map_city_roads_with_depot_interactive",
    "map_services_interactive",
    "plot_benchmark_on_map",
    "plot_city_roads",
    "plot_city_roads_with_depot",
    "plot_services_on_map",
    "prepare_city_road_network",
    "print_feasibility_report",
    "save_instance_overview_map",
    "save_road_network_figure",
    "save_road_network_with_markers",
]

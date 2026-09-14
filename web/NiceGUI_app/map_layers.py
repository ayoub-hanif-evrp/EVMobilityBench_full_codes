"""
PyDeck helpers for the NiceGUI wizard.
Assets are loaded from web/NiceGUI_app/assets.
"""

from __future__ import annotations

import functools
import struct
import sys
from pathlib import Path
from typing import Any

import pydeck as pdk

_REPO_ROOT = Path(__file__).resolve().parents[2]
_src = str(_REPO_ROOT / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from evrp_instance_generator_framework.types import depot_facility_latlon, primary_depot_facility_latlon

def assets_dir() -> Path:
    return Path(__file__).resolve().parent / "assets"


_DEPOT_MAP_COLOR = [255, 30, 30]

_SPLIT_MARKER_FILES = {
    "depot": "depot.png",
    "customer": "customer.png",
    "station": "charging_station.png",
    "satellite": "sub_depot.png",
}

_ICON_PX_DEPOT_STEP = 30
_ICON_PX_DEPOT_STEP_EXTRA = 28
_ICON_PX_RESULT_CUSTOMER = 32
_ICON_PX_RESULT_STATION = 34
_ICON_PX_RESULT_DEPOT = 34
_ICON_PX_RESULT_SATELLITE = 32

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


def _highway_key(data: dict) -> str:
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


def _highway_color_width(data: dict) -> tuple[list[int], float]:
    key = _highway_key(data)
    pair = _HIGHWAY_STYLE.get(key)
    if pair is not None:
        return pair[0], pair[1]
    for prefix in _HIGHWAY_RANK:
        if key.startswith(prefix):
            return _HIGHWAY_STYLE.get(prefix, ([155, 155, 155], 1.4))
    return [125, 120, 115], 1.4


def highway_color_for_key(highway_key: str) -> list[int]:
    """Expose road class color for legend rendering."""
    return _highway_color_width({"highway": highway_key})[0]


def edge_to_path_item(G, u, v, data):
    xu, yu = float(G.nodes[u]["x"]), float(G.nodes[u]["y"])
    xv, yv = float(G.nodes[v]["x"]), float(G.nodes[v]["y"])
    geom = data.get("geometry")
    if geom is not None and hasattr(geom, "coords"):
        path = [[float(x), float(y)] for (x, y) in geom.coords]
        if path:
            path[0] = [xu, yu]
            path[-1] = [xv, yv]
    else:
        path = [[xu, yu], [xv, yv]]
    color, width = _highway_color_width(data)
    return {"path": path, "color": color, "width": width}


def get_map_center(G):
    lats = [float(d["y"]) for _, d in G.nodes(data=True)]
    lons = [float(d["x"]) for _, d in G.nodes(data=True)]
    return sum(lats) / len(lats), sum(lons) / len(lons)


def graph_bbox_wsen(G):
    lats = [float(d["y"]) for _, d in G.nodes(data=True)]
    lons = [float(d["x"]) for _, d in G.nodes(data=True)]
    return min(lons), min(lats), max(lons), max(lats)


def latlon_in_graph_bbox(lat, lon, bbox, pad_deg=0.0008):
    west, south, east, north = bbox
    return (south - pad_deg) <= lat <= (north + pad_deg) and (west - pad_deg) <= lon <= (east + pad_deg)


@functools.lru_cache(maxsize=8)
def _marker_png_icon_pack(filename: str) -> tuple[str, dict[str, dict]] | None:
    src = (assets_dir() / filename).resolve()
    if not src.is_file():
        return None
    hdr = src.read_bytes()[:24]
    if len(hdr) < 24 or hdr[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    w, h = struct.unpack(">II", hdr[16:24])
    mmap = {"m": {"x": 0, "y": 0, "width": w, "height": h, "anchorY": h, "anchorX": w // 2}}
    return str(src), mmap


def _rows_with_sprite_key(rows: list[dict]) -> list[dict]:
    return [{**r, "ic": "m"} for r in rows]


def legend_icon_path(kind: str) -> str | None:
    fn = _SPLIT_MARKER_FILES.get(kind)
    if not fn:
        return None
    p = (assets_dir() / fn).resolve()
    return str(p) if p.is_file() else None


def highway_legend_keys(G) -> list[str]:
    """Pick common OSM road classes for compact map legends."""
    counts: dict[str, int] = {}
    for _, _, d in G.edges(data=True):
        k = _highway_key(d)
        counts[k] = counts.get(k, 0) + 1
    if not counts:
        return []
    n_e = sum(counts.values())
    floor = max(30, min(400, int(0.0015 * max(n_e, 1))))
    frequent = {k for k, c in counts.items() if c >= floor}
    if not frequent:
        top = sorted(counts.items(), key=lambda x: -x[1])[:8]
        frequent = {k for k, _ in top}
    ordered: list[str] = []
    remaining = set(frequent)
    for hw in _HIGHWAY_RANK:
        if hw in remaining:
            ordered.append(hw)
            remaining.discard(hw)
    for hw in sorted(remaining):
        ordered.append(hw)
    return ordered


def facility_pins_preview(
    *,
    n_total: int,
    center_lat: float,
    center_lon: float,
    lats: list[float],
    lons: list[float],
    node_ids: list[int | None],
    primary_depot_node_id: int | None,
) -> list[dict[str, Any]]:
    pins = []
    for i in range(n_total):
        lat = float(lats[i]) if i < len(lats) else center_lat
        lon = float(lons[i]) if i < len(lons) else center_lon
        nid = node_ids[i] if i < len(node_ids) else None
        if nid is None and i == 0 and primary_depot_node_id is not None:
            nid = int(primary_depot_node_id)
        pins.append({"lat": lat, "lon": lon, "node_id": nid, "color": _DEPOT_MAP_COLOR, "r": 24})
    return pins


def depot_step_pydeck_layers(
    G,
    depot_node_id=None,
    *,
    facility_pins: list | None = None,
    facility_pin_style: str | None = None,
):
    edges_list = list(G.edges(data=True))
    edge_items = [edge_to_path_item(G, u, v, d) for u, v, d in edges_list]
    hide_nodes: set = set()
    if depot_node_id is not None:
        hide_nodes.add(int(depot_node_id))
    if facility_pins:
        for p in facility_pins:
            nid = p.get("node_id")
            if nid is not None:
                hide_nodes.add(int(nid))
    nodes = []
    for n, d in G.nodes(data=True):
        if n in hide_nodes:
            continue
        nodes.append({"node_id": n, "lat": float(d["y"]), "lon": float(d["x"])})
    layers = [
        pdk.Layer(
            "PathLayer",
            id="depot_roads",
            data=edge_items,
            get_path="path",
            get_color="color",
            get_width="width",
            width_min_pixels=1.25,
            pickable=False,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            id="depot_intersections",
            data=nodes,
            get_position="[lon, lat]",
            get_fill_color=[100, 140, 255],
            get_radius=14,
            pickable=True,
            auto_highlight=True,
        ),
    ]
    if facility_pins:
        rows: list[dict] = []
        for i, p in enumerate(facility_pins):
            use_sat = facility_pin_style == "two_echelon" and i >= 1
            label = (
                f"Satellite hub {i}"
                if use_sat
                else ("Primary depot" if i == 0 else f"Depot {i + 1}")
            )
            rows.append(
                {
                    "lat": float(p["lat"]),
                    "lon": float(p["lon"]),
                    "node_id": p.get("node_id"),
                    "icon": "satellite" if use_sat else "depot",
                    "label": label,
                }
            )
        has_sat_pins = any(r["icon"] == "satellite" for r in rows)
        dep_fn = assets_dir() / _SPLIT_MARKER_FILES["depot"]
        sat_fn = assets_dir() / _SPLIT_MARKER_FILES["satellite"]
        use_split = dep_fn.is_file() and (sat_fn.is_file() or not has_sat_pins)
        if use_split:
            dpack = _marker_png_icon_pack(_SPLIT_MARKER_FILES["depot"])
            dep_rows = _rows_with_sprite_key([r for r in rows if r["icon"] == "depot"])
            if dpack and dep_rows:
                layers.append(
                    pdk.Layer(
                        "IconLayer",
                        id="depot_facilities_depot",
                        data=dep_rows,
                        get_position="[lon, lat]",
                        get_icon="ic",
                        icon_atlas=dpack[0],
                        icon_mapping=dpack[1],
                        size_scale=1,
                        get_size=_ICON_PX_DEPOT_STEP,
                        pickable=True,
                    )
                )
            if has_sat_pins:
                spack = _marker_png_icon_pack(_SPLIT_MARKER_FILES["satellite"])
                sat_rows = _rows_with_sprite_key([r for r in rows if r["icon"] == "satellite"])
                if spack and sat_rows:
                    layers.append(
                        pdk.Layer(
                            "IconLayer",
                            id="depot_facilities_sat",
                            data=sat_rows,
                            get_position="[lon, lat]",
                            get_icon="ic",
                            icon_atlas=spack[0],
                            icon_mapping=spack[1],
                            size_scale=1,
                            get_size=_ICON_PX_DEPOT_STEP,
                            pickable=True,
                        )
                    )
                elif sat_rows:
                    layers.append(
                        pdk.Layer(
                            "ScatterplotLayer",
                            id="depot_facilities_sat_sc",
                            data=sat_rows,
                            get_position="[lon, lat]",
                            get_fill_color=[160, 32, 240],
                            get_radius=22,
                            pickable=True,
                        )
                    )
        else:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    id="depot_facilities",
                    data=facility_pins,
                    get_position="[lon, lat]",
                    get_fill_color="color",
                    get_radius="r",
                    pickable=True,
                    auto_highlight=True,
                )
            )
    elif depot_node_id is not None and depot_node_id in G.nodes:
        d = G.nodes[depot_node_id]
        _dep_asset = assets_dir() / _SPLIT_MARKER_FILES["depot"]
        pin_row = {
            "lat": float(d["y"]),
            "lon": float(d["x"]),
            "node_id": str(depot_node_id),
            "label": "Depot",
        }
        dpack = _marker_png_icon_pack(_SPLIT_MARKER_FILES["depot"])
        if dpack and _dep_asset.is_file():
            layers.append(
                pdk.Layer(
                    "IconLayer",
                    id="depot_pin",
                    data=_rows_with_sprite_key([pin_row]),
                    get_position="[lon, lat]",
                    get_icon="ic",
                    icon_atlas=dpack[0],
                    icon_mapping=dpack[1],
                    size_scale=1,
                    get_size=_ICON_PX_DEPOT_STEP,
                    pickable=True,
                )
            )
        else:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    id="depot_pin",
                    data=[pin_row],
                    get_position="[lon, lat]",
                    get_fill_color=_DEPOT_MAP_COLOR,
                    get_radius=24,
                    pickable=True,
                    auto_highlight=True,
                )
            )
    return layers


def road_overview_layers(G):
    """Road-only map for Step 1 (no depot/facility pins)."""
    edges_list = list(G.edges(data=True))
    edge_items = [edge_to_path_item(G, u, v, d) for u, v, d in edges_list]
    nodes = [{"node_id": n, "lat": float(d["y"]), "lon": float(d["x"])} for n, d in G.nodes(data=True)]
    return [
        pdk.Layer(
            "PathLayer",
            id="overview_roads",
            data=edge_items,
            get_path="path",
            get_color="color",
            get_width="width",
            width_min_pixels=1.2,
            pickable=False,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            id="overview_nodes",
            data=nodes,
            get_position="[lon, lat]",
            get_fill_color=[100, 140, 255],
            get_radius=10,
            pickable=False,
        ),
    ]


def results_pydeck_layers(instance, Gi):
    depot_pts: list = []
    if getattr(instance, "depots", None) and len(instance.depots) > 0:
        for dep in instance.depots:
            la, lo = depot_facility_latlon(dep)
            depot_pts.append(
                {
                    "lat": la,
                    "lon": lo,
                    "label": f"Depot {dep.id + 1}",
                    "color": _DEPOT_MAP_COLOR,
                    "r": 52,
                }
            )
        d_lat = sum(p["lat"] for p in depot_pts) / len(depot_pts)
        d_lon = sum(p["lon"] for p in depot_pts) / len(depot_pts)
    else:
        d_lat, d_lon = primary_depot_facility_latlon(instance)
        depot_pts = [
            {
                "lat": d_lat,
                "lon": d_lon,
                "label": "Depot (facility)",
                "color": _DEPOT_MAP_COLOR,
                "r": 52,
            },
        ]

    edge_items = [edge_to_path_item(Gi, u, v, d) for u, v, d in Gi.edges(data=True)]
    road_pts = [{"lat": float(d["y"]), "lon": float(d["x"])} for _, d in Gi.nodes(data=True)]
    cust_pts = [{"lat": c.lat, "lon": c.lon, "label": f"C{c.id} demand={c.demand}"} for c in instance.customers]
    stat_pts = [
        {"lat": s.lat, "lon": s.lon, "label": f"S{s.id} {s.station_type} {s.charging_power_kW}kW"}
        for s in instance.stations
    ]

    base = assets_dir()
    _has_sats = bool(getattr(instance, "satellites", None) and instance.satellites)
    _split_step5 = (
        (base / _SPLIT_MARKER_FILES["customer"]).is_file()
        and (base / _SPLIT_MARKER_FILES["station"]).is_file()
        and (base / _SPLIT_MARKER_FILES["depot"]).is_file()
        and (not _has_sats or (base / _SPLIT_MARKER_FILES["satellite"]).is_file())
    )
    map_layers = [
        pdk.Layer(
            "PathLayer",
            data=edge_items,
            get_path="path",
            get_color="color",
            get_width="width",
            width_min_pixels=1.25,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=road_pts,
            get_position="[lon, lat]",
            get_fill_color=[190, 215, 255],
            get_radius=4,
        ),
    ]

    def _append_split_icon(layer_id: str, pts: list, filename: str, px: int, *, highlight: bool = False):
        if not pts:
            return
        pack = _marker_png_icon_pack(filename)
        if not pack:
            return
        kw = dict(
            id=layer_id,
            data=_rows_with_sprite_key(pts),
            get_position="[lon, lat]",
            get_icon="ic",
            icon_atlas=pack[0],
            icon_mapping=pack[1],
            size_scale=1,
            get_size=px,
            pickable=True,
        )
        if highlight:
            kw["auto_highlight"] = True
        map_layers.append(pdk.Layer("IconLayer", **kw))

    if _split_step5:
        _append_split_icon("res_customers", cust_pts, _SPLIT_MARKER_FILES["customer"], _ICON_PX_RESULT_CUSTOMER)
        _append_split_icon("res_stations", stat_pts, _SPLIT_MARKER_FILES["station"], _ICON_PX_RESULT_STATION)
        _append_split_icon(
            "instance_depots",
            depot_pts,
            _SPLIT_MARKER_FILES["depot"],
            _ICON_PX_RESULT_DEPOT,
            highlight=True,
        )
    else:
        map_layers.extend(
            [
                pdk.Layer(
                    "ScatterplotLayer",
                    data=cust_pts,
                    get_position="[lon, lat]",
                    get_fill_color=[255, 140, 0],
                    get_radius=35,
                    pickable=True,
                ),
                pdk.Layer(
                    "ScatterplotLayer",
                    data=stat_pts,
                    get_position="[lon, lat]",
                    get_fill_color=[0, 180, 80],
                    get_radius=45,
                    pickable=True,
                ),
                pdk.Layer(
                    "ScatterplotLayer",
                    id="instance_depots",
                    data=depot_pts,
                    get_position="[lon, lat]",
                    get_fill_color="color",
                    get_radius="r",
                    pickable=True,
                    auto_highlight=True,
                ),
            ]
        )

    if _has_sats:
        sat_pts = [
            {"lat": s.lat, "lon": s.lon, "label": f"Sat {s.id} cap={s.capacity}"}
            for s in instance.satellites
        ]
        if _split_step5:
            _append_split_icon(
                "res_satellites",
                sat_pts,
                _SPLIT_MARKER_FILES["satellite"],
                _ICON_PX_RESULT_SATELLITE,
            )
        else:
            map_layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=sat_pts,
                    get_position="[lon, lat]",
                    get_fill_color=[160, 32, 240],
                    get_radius=50,
                    pickable=True,
                )
            )

    return map_layers, d_lat, d_lon

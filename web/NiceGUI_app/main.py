"""
NiceGUI web interface for configuring and generating EVMobilityBench instances.
Run from repo root: python web/NiceGUI_app/main.py
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import os
import pkgutil
import sys
from pathlib import Path

import osmnx as ox
import pydeck as pdk

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_APP_DIR))

from evrp_instance_generator_framework.data.osm_disk_cache import OsmDiskCache, default_cache_dir
from evrp_instance_generator_framework.export.instance_export import EXPORT_DESCRIPTIONS, export_instance, export_keys_for_variant
from evrp_instance_generator_framework.road_network.osm_graph_download import download_directed_drive_graph
from evrp_instance_generator_framework.road_network.strongly_connected_component import largest_strongly_connected_component
from evrp_instance_generator_framework.road_network.utils import depot_single_source_times
from evrp_instance_generator_framework.variants.multi_depot import suggest_additional_depot_facilities
from evrp_instance_generator_framework.variants.two_echelon import suggest_satellite_facility_latlons

from evrp_instance_generator_framework.types import GenerationConfig as _GC

from generation import run_generation
from map_layers import (
    depot_step_pydeck_layers,
    facility_pins_preview,
    get_map_center,
    graph_bbox_wsen,
    highway_color_for_key,
    highway_legend_keys,
    latlon_in_graph_bbox,
    legend_icon_path,
    road_overview_layers,
    results_pydeck_layers,
)

# Python 3.14 removed pkgutil.find_loader; older NiceGUI deps still expect it.
if not hasattr(pkgutil, "find_loader"):
    pkgutil.find_loader = lambda name: importlib.util.find_spec(name)

from nicegui import app, run, ui
from state import MapEmbed, WizardState

VARIANT_LABELS = {
    "classic_evrptw": "Classic EVRPTW",
    "multi_depot_evrptw": "Multi-Depot EVRPTW",
    "two_echelon_evrp": "Two-Echelon EVRP",
}

STEP_LABELS = [
    "1 · Road network",
    "2 · Variant & params",
    "3 · Depot(s)",
    "4 · EV features",
    "5 · Results & map",
    "6 · Export",
]

STEP_HINTS = [
    "Download the OSM drive network for your city or region.",
    "Choose the variant, customer settings, traffic period, and feasibility scope.",
    "Place facilities on the map and snap each one to a road node.",
    "Configure battery, vehicle mass, aerodynamics, and climate effects for the energy model.",
    "Review routes, customers, stations, and feasibility on the map.",
    "Select components and format, then export your benchmark bundle.",
]

ws = WizardState()


def _set_ws_int(attr: str, value) -> None:
    if value is not None:
        setattr(ws, attr, int(value))


def _set_ws_float(attr: str, value) -> None:
    if value is not None:
        setattr(ws, attr, float(value))


def _set_ws_int_min(attr: str, value, minimum: int = 1) -> None:
    if value is not None:
        setattr(ws, attr, max(minimum, int(value)))


def header_logo_path() -> str | None:
    p = (_REPO / "logo_evrp.png").resolve()
    return str(p) if p.is_file() else None


def header_logo_browser_url() -> str | None:
    p = header_logo_path()
    if not p:
        return None
    ext = Path(p).suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
        ".webp": "image/webp",
    }.get(ext, "image/png")
    b64 = base64.b64encode(Path(p).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


_ASSET_ROUTES_REGISTERED = False


def register_asset_static_routes() -> None:
    global _ASSET_ROUTES_REGISTERED
    if _ASSET_ROUTES_REGISTERED:
        return
    (_APP_DIR / "assets").mkdir(parents=True, exist_ok=True)
    (_APP_DIR / "map_cache").mkdir(parents=True, exist_ok=True)
    app.add_static_files("/evrp-assets", str((_APP_DIR / "assets").resolve()))
    app.add_static_files("/evrp-maps", str((_APP_DIR / "map_cache").resolve()))
    _ASSET_ROUTES_REGISTERED = True


def show_map_embed(view: MapEmbed) -> None:
    """Render a PyDeck map via same-origin iframe (works on all NiceGUI versions)."""
    if not view:
        return
    ui.element("iframe").props(
        f'src="{view.src}" sandbox="allow-scripts allow-same-origin allow-popups allow-forms"'
    ).style(
        f"display:block;width:100%;height:{view.height_px}px;border:none;border-radius:12px;"
        f"box-shadow:0 4px 24px rgba(15,23,42,0.08);"
    )


def deck_to_map_embed(deck: pdk.Deck, name: str, height_px: int = 520) -> MapEmbed:
    """Write PyDeck HTML to disk and return iframe src (avoids huge inline HTML blobs)."""
    cache_dir = _APP_DIR / "map_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    filename = f"{safe}.html"
    path = cache_dir / filename
    deck.to_html(
        str(path),
        notebook_display=False,
        iframe_width="100%",
        iframe_height=height_px,
    )
    version = int(path.stat().st_mtime_ns)
    return MapEmbed(src=f"/evrp-maps/{filename}?v={version}", height_px=height_px)


def set_status(message: str = "", level: str = "info") -> None:
    ws.status_message = message
    ws.status_level = level


def map_legend(two_echelon: bool = False) -> None:
    with ui.expansion("Legend", icon="legend_toggle").classes("w-full border border-slate-200 rounded-lg bg-white/90"):
        with ui.row().classes("w-full gap-10 items-start"):
            with ui.column().classes("gap-2 min-w-[280px]"):
                ui.label("Road lines (common OSM types only)").classes("text-xs uppercase tracking-wide text-slate-500")
                if ws.movement_graph is not None:
                    for hw in highway_legend_keys(ws.movement_graph):
                        rgb = highway_color_for_key(hw)
                        ui.html(
                            f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;'
                            f'background:rgb({rgb[0]},{rgb[1]},{rgb[2]});margin-right:8px;border:1px solid #cbd5e1;vertical-align:middle;"></span>'
                            f'{hw.replace("_", " ").title()}'
                        ).classes("text-sm text-slate-700")
            with ui.column().classes("gap-2 min-w-[280px]"):
                ui.label("Facilities (map)").classes("text-xs uppercase tracking-wide text-slate-500")
                with ui.row().classes("items-center gap-2"):
                    p = legend_icon_path("depot")
                    if p:
                        ui.image(p).classes("w-5 h-5")
                    else:
                        ui.html('<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#ef4444;border:1px solid #b91c1c;"></span>')
                    ui.label("Primary depot / depot pins").classes("text-sm text-slate-700")
                if two_echelon:
                    with ui.row().classes("items-center gap-2"):
                        p = legend_icon_path("satellite")
                        if p:
                            ui.image(p).classes("w-5 h-5")
                        else:
                            ui.html('<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#a855f7;border:1px solid #7e22ce;"></span>')
                        ui.label("Satellite hubs").classes("text-sm text-slate-700")
        ui.label(
            "Bluish lines (not dots) are mostly residential/living_street in OSM here. Bright blue dots are road junctions."
        ).classes("text-xs text-slate-500 mt-2")


def road_map_legend() -> None:
    with ui.expansion("Legend", icon="legend_toggle").classes("w-full border border-slate-200 rounded-lg bg-white/90"):
        with ui.column().classes("gap-2"):
            ui.label("Road lines (common OSM types only)").classes("text-xs uppercase tracking-wide text-slate-500")
            if ws.movement_graph is not None:
                for hw in highway_legend_keys(ws.movement_graph):
                    rgb = highway_color_for_key(hw)
                    ui.html(
                        f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;'
                        f'background:rgb({rgb[0]},{rgb[1]},{rgb[2]});margin-right:8px;border:1px solid #cbd5e1;vertical-align:middle;"></span>'
                        f'{hw.replace("_", " ").title()}'
                    ).classes("text-sm text-slate-700")


def deck_to_iframe_html(deck: pdk.Deck, height_px: int = 520, *, name: str = "map") -> MapEmbed:
    return deck_to_map_embed(deck, name, height_px)


def update_depot_map(ws: WizardState):
    if ws.movement_graph is None:
        ws.map_embed_depot = MapEmbed()
        return
    c_lat, c_lon = get_map_center(ws.movement_graph)
    ws.ensure_facility_arrays(c_lat, c_lon)
    n = ws.n_facilities()
    is_multi = ws.variant == "multi_depot_evrptw"
    is_2e = ws.variant == "two_echelon_evrp"
    pins = facility_pins_preview(
        n_total=n,
        center_lat=c_lat,
        center_lon=c_lon,
        lats=ws.md_depot_lat,
        lons=ws.md_depot_lon,
        node_ids=ws.md_depot_node,
        primary_depot_node_id=ws.depot_node_id,
    )
    pin_style = "two_echelon" if is_2e else ("multi_depot" if is_multi else None)
    if is_multi or is_2e:
        mlats = [ws.md_depot_lat[i] if i < len(ws.md_depot_lat) else c_lat for i in range(n)]
        mlons = [ws.md_depot_lon[i] if i < len(ws.md_depot_lon) else c_lon for i in range(n)]
        map_lat = sum(mlats) / max(len(mlats), 1)
        map_lon = sum(mlons) / max(len(mlons), 1)
    elif ws.depot_node_id is not None:
        nd = ws.movement_graph.nodes[ws.depot_node_id]
        map_lat, map_lon = float(nd["y"]), float(nd["x"])
    else:
        map_lat, map_lon = float(ws.md_depot_lat[0]), float(ws.md_depot_lon[0])

    layers = depot_step_pydeck_layers(
        ws.movement_graph,
        ws.depot_node_id,
        facility_pins=pins,
        facility_pin_style=pin_style,
    )

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=map_lat, longitude=map_lon, zoom=14, pitch=0),
        map_style="light",
        map_provider="carto",
        tooltip={"text": "{label}\nnode {node_id}\nlat {lat}\nlon {lon}"},
    )
    ws.map_embed_depot = deck_to_map_embed(deck, "depot", 620)


def update_overview_map(ws: WizardState) -> None:
    if ws.movement_graph is None:
        ws.map_embed_overview = MapEmbed()
        return
    west, south, east, north = graph_bbox_wsen(ws.movement_graph)
    span = max(east - west, north - south)
    if span > 1.0:
        zoom = 9
    elif span > 0.45:
        zoom = 10
    elif span > 0.22:
        zoom = 11
    else:
        zoom = 12
    c_lat, c_lon = get_map_center(ws.movement_graph)
    deck = pdk.Deck(
        layers=road_overview_layers(ws.movement_graph),
        initial_view_state=pdk.ViewState(latitude=c_lat, longitude=c_lon, zoom=zoom, pitch=0),
        map_style="light",
        map_provider="carto",
    )
    ws.map_embed_overview = deck_to_map_embed(deck, "overview", 620)


def update_result_map(ws: WizardState):
    if ws.instance is None:
        ws.map_embed_result = MapEmbed()
        return
    Gi = ws.instance.movement_graph
    layers, d_lat, d_lon = results_pydeck_layers(ws.instance, Gi)
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=d_lat, longitude=d_lon, zoom=12, pitch=0),
        map_style="light",
        map_provider="carto",
        tooltip={"text": "{label}"},
    )
    ws.map_embed_result = deck_to_map_embed(deck, "result", 520)


def sync_load_graph(ws: WizardState) -> str | None:
    try:
        _dc = OsmDiskCache(default_cache_dir())
        G = download_directed_drive_graph(ws.city, ws.country, disk_cache=_dc)
        G = largest_strongly_connected_component(G)
        if "crs" not in G.graph:
            G.graph["crs"] = "EPSG:4326"
        ws.movement_graph = G
        c_lat, c_lon = get_map_center(G)
        ws.md_depot_lat = [c_lat]
        ws.md_depot_lon = [c_lon]
        ws.md_depot_node = [None]
        ws.depot_node_id = None
        ws.depot_facility_lat = None
        ws.depot_facility_lon = None
        ws.instance = None
        ws.extra_depots = []
        update_overview_map(ws)
        update_depot_map(ws)
        return None
    except Exception as exc:
        return str(exc)


def snap_facility(ws: WizardState, index: int, *, update_map: bool = True) -> str | None:
    G = ws.movement_graph
    if G is None:
        return "No graph"
    bbox = graph_bbox_wsen(G)
    la = float(ws.md_depot_lat[index])
    lo = float(ws.md_depot_lon[index])
    if not latlon_in_graph_bbox(la, lo, bbox):
        return "Coordinates outside road network bbox"
    nid = int(ox.distance.nearest_nodes(G, X=lo, Y=la))
    ws.md_depot_node[index] = nid
    nd = G.nodes[nid]
    ws.md_depot_lat[index] = float(nd["y"])
    ws.md_depot_lon[index] = float(nd["x"])
    if index == 0:
        ws.depot_node_id = nid
        ws.depot_facility_lat = float(la)
        ws.depot_facility_lon = float(lo)
    sync_extra_depots(ws)
    if update_map:
        update_depot_map(ws)
    return None


def facility_snap_busy_message(ws: WizardState, idx: int) -> str:
    if ws.variant == "multi_depot_evrptw":
        return "Snapping primary depot…" if idx == 0 else f"Snapping depot {idx + 1}…"
    if ws.variant == "two_echelon_evrp":
        return "Snapping central depot…" if idx == 0 else f"Snapping satellite {idx}…"
    return "Snapping depot…"


def sync_extra_depots(ws: WizardState):
    if ws.variant == "multi_depot_evrptw":
        n = ws.n_facilities()
        ws.extra_depots = [
            (float(ws.md_depot_lat[j]), float(ws.md_depot_lon[j]))
            for j in range(1, n)
            if j < len(ws.md_depot_lat) and j < len(ws.md_depot_lon)
        ]


def generate_depots_osm(ws: WizardState) -> str | None:
    G = ws.movement_graph
    if G is None or not ws.city or not ws.country:
        return "Need city, country, and loaded graph"
    c_lat, c_lon = get_map_center(G)
    n = ws.n_facilities()
    cp = ws.cust_params_dict()
    _dc = OsmDiskCache(default_cache_dir())
    try:
        if ws.variant == "multi_depot_evrptw":
            n_add = max(0, n - 1)
            sugg = suggest_additional_depot_facilities(
                city=ws.city,
                country=ws.country,
                movement_graph=G,
                primary_lat=float(c_lat),
                primary_lon=float(c_lon),
                num_additional_depots=n_add,
                anchor_period=str(cp.get("energy_period", "off_peak")),
                snap_max_m=ws.default_depot_snap_m(),
                disk_cache=_dc,
            )
            ws.md_depot_lat[0], ws.md_depot_lon[0] = c_lat, c_lon
            n0 = int(ox.distance.nearest_nodes(G, X=c_lon, Y=c_lat))
            ws.md_depot_node[0] = n0
            ws.depot_node_id = n0
            ws.depot_facility_lat, ws.depot_facility_lon = float(c_lat), float(c_lon)
            for j in range(1, n):
                if j - 1 < len(sugg):
                    lat_i, lon_i = float(sugg[j - 1][0]), float(sugg[j - 1][1])
                else:
                    lat_i, lon_i = c_lat, c_lon
                ws.md_depot_lat[j] = lat_i
                ws.md_depot_lon[j] = lon_i
                ws.md_depot_node[j] = int(ox.distance.nearest_nodes(G, X=lon_i, Y=lat_i))
        elif ws.variant == "two_echelon_evrp":
            bbox = graph_bbox_wsen(G)
            n0 = int(ox.distance.nearest_nodes(G, X=c_lon, Y=c_lat))
            wt = f"{cp.get('energy_period', 'off_peak')}_travel_time_s"
            d_times = depot_single_source_times(G, n0, wt)
            from evrp_instance_generator_framework.types import GenerationConfig

            _sat_snap = float(GenerationConfig.__dataclass_fields__["satellite_snap_max_dist_m"].default)
            locs = suggest_satellite_facility_latlons(
                G,
                city=ws.city,
                country=ws.country,
                bbox=bbox,
                disk_cache=_dc,
                num_satellites=n - 1,
                primary_lat=float(c_lat),
                primary_lon=float(c_lon),
                depot_to_node_time=d_times,
                satellite_snap_max_dist_m=_sat_snap,
            )
            ws.md_depot_lat[0], ws.md_depot_lon[0] = c_lat, c_lon
            ws.md_depot_node[0] = n0
            ws.depot_node_id = n0
            ws.depot_facility_lat, ws.depot_facility_lon = float(c_lat), float(c_lon)
            for j, (la, lo) in enumerate(locs, start=1):
                if j >= n:
                    break
                ws.md_depot_lat[j] = float(la)
                ws.md_depot_lon[j] = float(lo)
                ws.md_depot_node[j] = int(ox.distance.nearest_nodes(G, X=float(lo), Y=float(la)))
        sync_extra_depots(ws)
        update_depot_map(ws)
        return None
    except Exception as exc:
        return str(exc)


def reset_depots_center(ws: WizardState):
    G = ws.movement_graph
    if G is None:
        return
    c_lat, c_lon = get_map_center(G)
    n = ws.n_facilities()
    ws.md_depot_lat = [c_lat] * n
    ws.md_depot_lon = [c_lon] * n
    ws.md_depot_node = [None] * n
    ws.depot_node_id = None
    ws.depot_facility_lat = None
    ws.depot_facility_lon = None
    ws.extra_depots = []
    update_depot_map(ws)


def all_depots_snapped(ws: WizardState) -> bool:
    G = ws.movement_graph
    if G is None:
        return False
    n = ws.n_facilities()
    for i in range(n):
        nid = ws.md_depot_node[i] if i < len(ws.md_depot_node) else None
        if nid is None or int(nid) not in G.nodes:
            return False
    return True


@ui.refreshable
def step_header():
    hint = STEP_HINTS[ws.step - 1]
    with ui.card().classes(
        "w-full overflow-hidden rounded-2xl shadow-lg border border-slate-200/70 "
        "bg-gradient-to-b from-white via-white to-slate-50/90 backdrop-blur-sm"
    ):
        ui.html(
            '<div style="width:100%;height:4px;background:linear-gradient(90deg,#34d399,#10b981,#6ee7b7);'
            'border-radius:14px 14px 0 0;"></div>'
        ).classes("w-full")
        with ui.column().classes("w-full items-center px-4 py-4 md:px-6 md:py-5 gap-3"):
            lu = header_logo_browser_url()
            if lu:
                ui.html(
                    f'<img src="{lu}" alt="EVRP logo" '
                    'style="display:block;width:100%;max-width:360px;height:auto;object-fit:contain;object-position:center;" />'
                ).classes("w-full flex justify-center")
            with ui.row().classes("w-full justify-center gap-2 flex-wrap"):
                for i, label in enumerate(STEP_LABELS, start=1):
                    if ws.step == i:
                        ui.badge(label, color="positive").classes(
                            "px-3 py-1.5 md:px-4 shadow-sm font-semibold text-xs "
                            "ring-2 ring-emerald-300/80 bg-emerald-100 text-emerald-950"
                        )
                    elif ws.step > i:
                        ui.badge(label, color="positive").props("outline").classes(
                            "px-3 py-1.5 opacity-95 font-medium text-xs border-emerald-300 text-emerald-800"
                        )
                    else:
                        ui.badge(label, color="grey-6").props("outline").classes(
                            "px-3 py-1.5 font-medium text-xs text-slate-500"
                        )
            ui.label(hint).classes(
                "text-center text-sm leading-relaxed text-slate-600 "
                "max-w-xl px-2 border-t border-slate-200/80 pt-3 mt-0.5"
            )


@ui.refreshable
def body_content():
    if ws.busy_message and ws.step != 5:
        with ui.card().classes("w-full border border-emerald-200 bg-emerald-50"):
            with ui.row().classes("items-center gap-3"):
                ui.spinner(size="sm", color="positive")
                ui.label(ws.busy_message).classes("text-emerald-900")

    if ws.step_transition_caption:
        with ui.column().classes("w-full items-center py-16"):
            with ui.card().classes(
                "w-full max-w-md rounded-2xl shadow-xl border border-emerald-100/90 "
                "bg-gradient-to-br from-white via-emerald-50/90 to-emerald-100/40 px-10 py-12"
            ):
                with ui.column().classes("items-center gap-4"):
                    ui.spinner(size="lg", color="positive")
                    ui.label("Opening next step").classes(
                        "text-xs font-semibold uppercase tracking-[0.18em] text-emerald-700/90"
                    )
                    ui.label(ws.step_transition_caption).classes(
                        "text-xl font-bold text-emerald-950 text-center tracking-tight"
                    )
                    ui.label("Just a moment").classes("text-sm text-slate-500 text-center")
        return

    if ws.status_message:
        color_map = {
            "error": "bg-red-50 border-red-200 text-red-700",
            "success": "bg-green-50 border-green-200 text-green-700",
            "warning": "bg-amber-50 border-amber-200 text-amber-700",
        }
        cls = color_map.get(ws.status_level, "bg-emerald-50 border-emerald-200 text-emerald-800")
        with ui.card().classes(f"w-full mb-4 border {cls}"):
            with ui.row().classes("items-center justify-between w-full"):
                ui.label(ws.status_message)
                ui.button(icon="close", on_click=lambda: (set_status(""), body_content.refresh())).props("flat round dense")
    if ws.step == 1:
        render_step1()
    elif ws.step == 2:
        render_step2()
    elif ws.step == 3:
        render_step3()
    elif ws.step == 4:
        render_step4()
    elif ws.step == 5:
        render_step5()
    else:
        render_step6()


def _apply_step(s: int) -> None:
    set_status("")
    ws.step_transition_caption = ""
    if ws.step == 3 and s != 3:
        ws.facility_snap_busy = ""
    ws.step = s


async def navigate_step(target: int) -> None:
    if target < 1 or target > len(STEP_LABELS):
        return
    caption = STEP_LABELS[target - 1]
    ws.step_transition_caption = caption
    body_content.refresh()
    await asyncio.sleep(0.16)
    _apply_step(target)
    step_header.refresh()
    body_content.refresh()


def step_nav_handler(target: int):
    async def _handler(*_: object) -> None:
        await navigate_step(target)

    return _handler


def go_step(s: int) -> None:
    """Instant step change without transition (internal / tests). Prefer navigate_step in UI."""
    _apply_step(s)
    step_header.refresh()
    body_content.refresh()


def set_variant_and_refresh(value: str) -> None:
    ws.variant = str(value)
    body_content.refresh()


def render_step1():
    with ui.card().classes("w-full p-6 shadow-lg rounded-xl border border-slate-200/80 bg-white/95"):
        ui.label("Step 1 – Load the road network").classes("text-lg font-semibold text-slate-700 mb-4")
        with ui.grid(columns=2).classes("w-full gap-4"):
            ui.input("City").bind_value(ws, "city")
            ui.input("Country").bind_value(ws, "country")
        with ui.row().classes("gap-3 mt-4"):
            async def load_click():
                ws.busy_message = "Downloading OSM graph…"
                set_status("")
                body_content.refresh()
                err = await run.io_bound(sync_load_graph, ws)
                ws.busy_message = ""
                if err:
                    set_status(err, "error")
                else:
                    set_status("", "info")
                body_content.refresh()

            ui.button("Download road network", on_click=load_click, icon="download").props("unelevated no-caps").classes(
                "bg-emerald-600 text-white px-6"
            )

    if ws.movement_graph is not None:
        with ui.card().classes("w-full p-5 shadow-md rounded-xl border border-slate-200 bg-white/95"):
            ui.label(
                f"Road network ready: {ws.movement_graph.number_of_nodes()} nodes, {ws.movement_graph.number_of_edges()} edges"
            ).classes("font-medium text-emerald-700 mb-2")
            if ws.map_embed_overview:
                show_map_embed(ws.map_embed_overview)
                road_map_legend()
            ui.button(
                "Next ▶ Choose variant",
                on_click=step_nav_handler(2),
                icon="arrow_forward",
            ).props("unelevated no-caps").classes("bg-emerald-700 text-white mt-4")


def render_step2():
    with ui.card().classes("w-full p-6 shadow-lg rounded-xl border border-slate-200/80 bg-white/95"):
        ui.label("Step 2 – Problem variant & parameters").classes("text-lg font-semibold mb-4")
        with ui.column().classes("gap-4 w-full max-w-5xl mx-auto"):
            with ui.card().classes("w-full p-4 border border-slate-100 shadow-sm"):
                ui.label("Problem variant").classes("text-slate-700 font-medium")
                ui.select(VARIANT_LABELS, value=ws.variant, on_change=lambda e: set_variant_and_refresh(e.value)).props(
                    "label=EVRP variant outlined"
                )
            with ui.card().classes("w-full p-4 border border-slate-100 shadow-sm"):
                ui.label("Seed, traffic & infrastructure").classes("text-slate-700 font-medium mb-2")
                with ui.grid(columns=4).classes("w-full gap-4"):
                    ui.number("Seed", format="%.0f", value=ws.seed, on_change=lambda e: _set_ws_int("seed", e.value))
                    ui.number("Stations", format="%.0f", value=ws.num_stations, on_change=lambda e: _set_ws_int("num_stations", e.value))
                    ui.select(["off_peak", "midday", "pm_peak"], value=ws.energy_period, on_change=lambda e: setattr(ws, "energy_period", e.value)).props("label=Traffic period")
                    ui.checkbox("SRTM elevation", value=ws.use_elevation, on_change=lambda e: setattr(ws, "use_elevation", e.value))
            with ui.card().classes("w-full p-4 border border-slate-100 shadow-sm"):
                ui.label("Customers").classes("text-slate-700 font-medium mb-2")
                with ui.grid(columns=3).classes("w-full gap-4"):
                    ui.number("Customers", value=ws.num_customers, format="%.0f", on_change=lambda e: _set_ws_int("num_customers", e.value))
                    ui.number("Clusters", value=ws.num_clusters, format="%.0f", on_change=lambda e: _set_ws_int("num_clusters", e.value))
                    ui.select(["c", "r", "rc"], value=ws.pattern, on_change=lambda e: setattr(ws, "pattern", e.value)).props("label=Pattern")
                with ui.grid(columns=3).classes("w-full gap-4 mt-2"):
                    ui.number("Demand min", value=ws.demand_min, format="%.0f", on_change=lambda e: _set_ws_int("demand_min", e.value))
                    ui.number("Demand max", value=ws.demand_max, format="%.0f", on_change=lambda e: _set_ws_int("demand_max", e.value))
                    ui.select(["wide", "medium", "tight"], value=ws.tw_tightness, on_change=lambda e: setattr(ws, "tw_tightness", e.value)).props("label=TW tightness")

            if ws.pattern in ("c", "rc"):
                with ui.expansion("Cluster radius & spacing", icon="tune").classes("w-full"):
                    ui.number(
                        "Cluster max radius (m)",
                        value=ws.cluster_max_radius_m,
                        on_change=lambda e: _set_ws_float("cluster_max_radius_m", e.value),
                    )
                    ui.number(
                        "Min distance between customers (m)",
                        value=ws.cluster_min_separation_m,
                        on_change=lambda e: _set_ws_float("cluster_min_separation_m", e.value),
                    )

            if ws.variant == "multi_depot_evrptw":
                ui.number(
                    "Additional depots (beyond primary)",
                    value=ws.num_additional_depots,
                    format="%.0f",
                    on_change=lambda e: _set_ws_int_min("num_additional_depots", e.value),
                )
            if ws.variant == "two_echelon_evrp":
                ui.number(
                    "Number of satellites",
                    value=ws.two_echelon_num_satellites,
                    format="%.0f",
                    on_change=lambda e: _set_ws_int("two_echelon_num_satellites", e.value),
                )
            with ui.card().classes("w-full p-4 border border-slate-100 shadow-sm"):
                ui.label("Data source & feasibility").classes("text-slate-700 font-medium mb-2")
                with ui.column().classes("gap-3 w-full"):
                    ui.upload(
                        label="Customer CSV (optional)",
                        auto_upload=True,
                        on_upload=lambda e: _on_csv_upload(e),
                    ).classes("w-full")
                    ui.select(
                        {"time_only": "Time only (fast)", "time_and_energy": "Time & energy (full)"},
                        value=ws.feasibility_scope,
                        on_change=lambda e: setattr(ws, "feasibility_scope", e.value),
                        label="Feasibility scope",
                    ).classes("w-full")

        with ui.row().classes("gap-3 mt-6"):
            ui.button("← Back", on_click=step_nav_handler(1)).props("outline no-caps")
            ui.button(
                "Next ▶ Configure facilities",
                on_click=_enter_step3_async,
                icon="place",
            ).props("unelevated no-caps").classes("bg-emerald-600 text-white")


def _on_csv_upload(e):
    ws.customer_csv_bytes = e.content.read()
    set_status("Customer CSV loaded.", "success")
    body_content.refresh()


def _snap_click_handler(idx: int):
    async def _handler(*_: object) -> None:
        await _snap_idx(idx)

    return _handler


async def _enter_step3_async():
    if ws.movement_graph is None:
        set_status("Load a road network first.", "warning")
        body_content.refresh()
        return
    c_lat, c_lon = get_map_center(ws.movement_graph)
    ws.ensure_facility_arrays(c_lat, c_lon)
    sync_extra_depots(ws)
    ws.busy_message = "Preparing facilities map…"
    body_content.refresh()
    try:
        update_depot_map(ws)
        set_status("", "info")
    except Exception as exc:
        # Still move to Step 3 so user can continue; show warning + map placeholder.
        ws.map_embed_depot = MapEmbed()
        set_status(f"Step 3 opened, but map could not render: {exc}", "warning")
    finally:
        ws.busy_message = ""
    await navigate_step(3)


def render_step3():
    G = ws.movement_graph
    if G is None:
        ui.label("Load a graph in Step 1.").classes("text-negative")
        return

    is_multi = ws.variant == "multi_depot_evrptw"
    is_2e = ws.variant == "two_echelon_evrp"
    n = ws.n_facilities()
    title = (
        "Step 3 – Place all depots"
        if is_multi
        else "Step 3 – Place central depot & satellites"
        if is_2e
        else "Step 3 – Place the depot"
    )
    with ui.card().classes("w-full p-6 shadow-lg rounded-xl border border-slate-200/80"):
        ui.label(title).classes("text-lg font-semibold mb-2")
        if is_2e:
            ui.label(
                "Set facility coordinates; use Snap so each site maps to a road node. "
                "Generate from OSM suggests satellite sites."
            ).classes("text-slate-600 text-sm mb-4")

        if is_multi:
            with ui.row().classes("w-full justify-center gap-3 mb-4"):
                ui.button(
                    "Generate all depot locations (OSM)",
                    on_click=lambda: _async_gen_depots(),
                    icon="auto_awesome",
                ).props("unelevated no-caps").classes("bg-emerald-600 text-white")
                ui.button("Reset all depots", on_click=lambda: reset_depots_center(ws) or refresh_step3_maps()).props(
                    "outline no-caps"
                )
        elif is_2e:
            with ui.row().classes("w-full justify-center gap-3 mb-4"):
                ui.button(
                    "Generate depot + satellites (OSM)",
                    on_click=lambda: _async_gen_depots(),
                    icon="auto_awesome",
                ).classes("bg-emerald-600 text-white").props("unelevated no-caps")
                ui.button(
                    "Reset all to map center",
                    on_click=lambda: reset_depots_center(ws) or refresh_step3_maps(),
                ).props("outline no-caps")

        def _facility_lat_change(idx: int, val):
            if val is not None:
                ws.md_depot_lat[idx] = float(val)
                sync_extra_depots(ws)

        def _facility_lon_change(idx: int, val):
            if val is not None:
                ws.md_depot_lon[idx] = float(val)
                sync_extra_depots(ws)

        with ui.column().classes("w-full items-center"):
            for i in range(n):
                label = (
                    f"Depot {i + 1}"
                    if is_multi
                    else ("Central depot" if i == 0 else f"Satellite {i}")
                    if is_2e
                    else "Depot (single)"
                )
                with ui.card().classes("w-full max-w-xl p-4 mb-2 border border-slate-100"):
                    ui.label(label).classes("font-medium text-slate-700")
                    with ui.row().classes("items-end gap-3 w-full flex-wrap"):
                        ui.number(
                            "Latitude",
                            format="%.6f",
                            value=float(ws.md_depot_lat[i]),
                            step=1e-5,
                            on_change=lambda e, idx=i: _facility_lat_change(idx, e.value),
                        )
                        ui.number(
                            "Longitude",
                            format="%.6f",
                            value=float(ws.md_depot_lon[i]),
                            step=1e-5,
                            on_change=lambda e, idx=i: _facility_lon_change(idx, e.value),
                        )

                        ui.button(
                            "Snap",
                            on_click=_snap_click_handler(i),
                        ).props("outline no-caps")

        map_slot()
        ui.button("Refresh map preview", on_click=lambda: refresh_step3_maps(), icon="refresh").props("outline no-caps").classes("mt-2")
        map_legend(two_echelon=is_2e)
        if is_multi or is_2e:
            if all_depots_snapped(ws):
                ui.label("All facilities snapped to the road network.").classes("text-positive font-medium mt-2")
        elif ws.depot_node_id is not None:
            nid = ws.depot_node_id
            nd = G.nodes[nid]
            ui.label(f"Depot snapped to node {nid} at ({float(nd['y']):.6f}, {float(nd['x']):.6f})").classes(
                "text-positive mt-2"
            )

        async def _next_from_depot():
            if is_multi or is_2e:
                if not all_depots_snapped(ws):
                    set_status("Snap every facility before continuing.", "warning")
                    body_content.refresh()
                    return
            elif ws.depot_node_id is None:
                set_status("Snap the depot before continuing.", "warning")
                body_content.refresh()
                return
            await navigate_step(4)

        with ui.row().classes("gap-3 mt-4"):
            ui.button("← Back", on_click=step_nav_handler(2)).props("outline no-caps")
            ui.button(
                "Next ▶ EV features",
                icon="arrow_forward",
                on_click=_next_from_depot,
            ).props("unelevated no-caps").classes("bg-emerald-700 text-white")


@ui.refreshable
def map_slot():
    if ws.facility_snap_busy:
        with ui.column().classes("w-full items-center justify-center rounded-2xl border border-emerald-200 bg-emerald-50/70 py-20"):
            ui.spinner(size="lg", color="positive")
            ui.label(ws.facility_snap_busy).classes("text-emerald-950 font-semibold text-center mt-4 tracking-tight")
        return
    if ws.map_embed_depot:
        show_map_embed(ws.map_embed_depot)


def refresh_step3_maps():
    update_depot_map(ws)
    map_slot.refresh()


async def _snap_idx(idx: int):
    def _snap_io():
        return snap_facility(ws, idx, update_map=False)

    ws.facility_snap_busy = facility_snap_busy_message(ws, idx)
    ws.map_embed_depot = MapEmbed()
    map_slot.refresh()
    body_content.refresh()
    err = await run.io_bound(_snap_io)
    ws.facility_snap_busy = ""
    if err:
        set_status(err, "error")
    else:
        set_status("", "info")
    await run.io_bound(update_depot_map, ws)
    map_slot.refresh()
    body_content.refresh()


async def _async_gen_depots():
    err = await run.io_bound(generate_depots_osm, ws)
    if err:
        set_status(err, "warning")
    else:
        set_status("", "info")
    map_slot.refresh()
    body_content.refresh()


async def run_instance_generation() -> None:
    ws.instance = None
    ws.busy_message = "Generating instance, please wait."
    set_status("", "info")
    if ws.step != 5:
        await navigate_step(5)
    else:
        body_content.refresh()
    inst, err = await run.io_bound(run_generation, ws)
    ws.busy_message = ""
    if err:
        set_status(err, "error")
        body_content.refresh()
        return
    ws.instance = inst
    update_result_map(ws)
    set_status("", "info")
    body_content.refresh()


def render_step4():
    with ui.card().classes("w-full p-6 shadow-lg rounded-xl border border-slate-200/80 bg-white/95"):
        with ui.column().classes("w-full max-w-3xl mx-auto items-center gap-4"):
            ui.label("Step 4 – EV features").classes("text-lg font-semibold text-slate-800 text-center")
            ui.label(
                "Configure the electric vehicle used for energy and feasibility models: battery size, mass, "
                "aerodynamics, and optional climate or weather effects. These values drive consumption on the road network you loaded."
            ).classes("text-slate-600 text-sm text-center leading-relaxed max-w-2xl")
            with ui.card().classes("w-full p-4 border border-slate-100 shadow-sm"):
                ui.label("Vehicle & energy model").classes("text-slate-700 font-medium mb-3 text-center w-full")
                with ui.grid(columns=2).classes("w-full gap-x-6 gap-y-3"):
                    ui.number("Battery (kWh)", value=ws.battery_kwh, on_change=lambda e: _set_ws_float("battery_kwh", e.value))
                    ui.number("Mass (kg)", value=ws.mass_kg, on_change=lambda e: _set_ws_float("mass_kg", e.value))
                    ui.number("Rolling resistance", value=ws.rolling_f, on_change=lambda e: _set_ws_float("rolling_f", e.value))
                    ui.number("Rotating mass factor", value=ws.mass_factor, on_change=lambda e: _set_ws_float("mass_factor", e.value))
                    ui.number("Drag Cd", value=ws.drag_cd, on_change=lambda e: _set_ws_float("drag_cd", e.value))
                    ui.number("Frontal area (m²)", value=ws.frontal_m2, on_change=lambda e: _set_ws_float("frontal_m2", e.value))
            with ui.card().classes("w-full p-4 border border-slate-100 shadow-sm"):
                ui.label("Driver & environment").classes("text-slate-700 font-medium mb-3 text-center w-full")
                ui.select(
                    ["passive", "aggressive"],
                    value=ws.driver_behavior,
                    on_change=lambda e: setattr(ws, "driver_behavior", e.value),
                ).props("label=Driver behavior outlined").classes("w-full")
                with ui.row().classes("w-full justify-center gap-8 flex-wrap mt-2"):
                    ui.checkbox("Heating", value=ws.heating, on_change=lambda e: setattr(ws, "heating", e.value))
                    ui.checkbox("Cooling", value=ws.cooling, on_change=lambda e: setattr(ws, "cooling", e.value))
                    ui.checkbox("Rain", value=ws.rain, on_change=lambda e: setattr(ws, "rain", e.value))
            with ui.row().classes("w-full justify-center gap-3 mt-2"):
                ui.button("← Back", on_click=step_nav_handler(3)).props("outline no-caps")
                ui.button(
                    "Generate instance ▶",
                    on_click=run_instance_generation,
                ).classes("bg-emerald-600 text-white").props("unelevated no-caps")


def render_step5():
    if ws.instance is None:
        if ws.busy_message:
            with ui.column().classes("w-full items-center py-10"):
                with ui.card().classes(
                    "rounded-2xl shadow-lg border border-emerald-100/90 "
                    "bg-gradient-to-br from-white via-emerald-50/80 to-emerald-100/50 px-10 py-8"
                ):
                    with ui.column().classes("items-center gap-4"):
                        ui.spinner(size="lg", color="positive").classes("text-emerald-600")
                        ui.label(ws.busy_message).classes(
                            "text-emerald-950 font-semibold text-lg tracking-tight text-center"
                        )
            return
        with ui.column().classes("w-full items-center gap-4 py-8"):
            with ui.card().classes("w-full max-w-lg p-6 border border-red-200 bg-red-50/80"):
                ui.label("Instance generation failed").classes("text-lg font-semibold text-red-800")
                if ws.status_message:
                    ui.label(ws.status_message).classes("text-red-700 text-sm mt-2 whitespace-pre-wrap")
                else:
                    ui.label("An unknown error occurred.").classes("text-red-700 text-sm mt-2")
            with ui.row().classes("gap-3"):
                ui.button("← Back to EV features", on_click=step_nav_handler(4)).props("outline no-caps")
                ui.button(
                    "Retry generation",
                    on_click=run_instance_generation,
                    icon="refresh",
                ).props("unelevated no-caps").classes("bg-emerald-600 text-white")
        return
    inst = ws.instance
    cp = ws.cust_params_dict()
    variant = cp.get("variant", "classic_evrptw")
    msg = f"Variant: {VARIANT_LABELS.get(variant, variant)} · TW: {cp.get('time_window_tightness', 'medium')}"
    if variant == "two_echelon_evrp" and getattr(inst, "satellites", None):
        msg += f" · Satellites: {len(inst.satellites)}"
    with ui.row().classes("w-full justify-center mb-4"):
        ui.label(msg).classes("text-slate-700 bg-slate-100/90 border border-slate-200 px-5 py-2 rounded-xl shadow-sm font-medium text-center")

    report = getattr(inst, "generation_report", None) or {}
    status = report.get("status") or inst.metadata.extra.get("acceptance_status", "unknown")
    if status == "accepted":
        ui.label("Instance accepted for export.").classes("text-emerald-700 font-medium text-center w-full mb-2")
    elif status == "rejected":
        ui.label(
            "Instance generated but marked rejected (feasibility or consistency checks). "
            "Export will write to a rejected/ folder."
        ).classes("text-amber-700 font-medium text-center w-full mb-2")
    if report:
        with ui.expansion("Generation report", icon="fact_check").classes("w-full mb-3"):
            ui.code(json.dumps(report, indent=2, default=str), language="json").classes("w-full max-h-72")

    if ws.map_embed_result:
        show_map_embed(ws.map_embed_result)
        map_legend(two_echelon=bool(getattr(inst, "satellites", None)))

    with ui.expansion(f"Customers ({len(inst.customers)})", icon="people").classes("w-full"):
        cols = [
            {"name": "id", "label": "id", "field": "id", "align": "left"},
            {"name": "lat", "label": "lat", "field": "lat", "align": "left"},
            {"name": "demand", "label": "demand", "field": "demand", "align": "left"},
        ]
        rows = [
            {
                "id": c.id,
                "lat": round(c.lat, 6),
                "demand": c.demand,
            }
            for c in inst.customers
        ]
        ui.table(columns=cols, rows=rows, row_key="id").classes("w-full")

    with ui.expansion(f"Stations ({len(inst.stations)})", icon="ev_station"):
        cols = [
            {"name": "id", "label": "id", "field": "id", "align": "left"},
            {"name": "type", "label": "type", "field": "type", "align": "left"},
            {"name": "power", "label": "kW", "field": "power", "align": "left"},
        ]
        rows = [
            {"id": s.id, "type": s.station_type, "power": s.charging_power_kW}
            for s in inst.stations
        ]
        ui.table(columns=cols, rows=rows, row_key="id").classes("w-full")

    with ui.expansion("Feasibility", icon="rule"):
        ui.code(json.dumps(inst.feasibility, indent=2, default=str), language="json").classes("w-full max-h-96")

    with ui.expansion("Metadata", icon="info"):
        meta = inst.metadata
        ui.code(
            json.dumps(
                {
                    "variant": meta.variant,
                    "city": meta.city,
                    "country": meta.country,
                    "seed": meta.seed,
                    "customer_count": meta.customer_count,
                    "depot_count": meta.depot_count,
                    "satellite_count": meta.satellite_count,
                },
                indent=2,
            ),
            language="json",
        )

    with ui.row().classes("gap-3 mt-4"):
        ui.button("← Back (re-configure)", on_click=_back_from_results_async).props("outline no-caps")
        ui.button(
            "Next ▶ Export",
            on_click=step_nav_handler(6),
        ).classes("bg-emerald-700 text-white").props("unelevated no-caps")


async def _back_from_results_async():
    ws.instance = None
    await navigate_step(4)


def render_step6():
    if ws.instance is None:
        ui.label("No instance — go back to Step 5.").classes("text-negative")
        ui.button("← Back", on_click=step_nav_handler(5))
        return
    variant = getattr(ws.instance.metadata, "variant", "classic_evrptw")
    keys = export_keys_for_variant(variant)

    with ui.card().classes("w-full p-6 shadow-lg rounded-xl"):
        ui.label("Step 6 – Export dataset").classes("text-lg font-semibold mb-4")
        for key in keys:
            ws.export_selection.setdefault(key, True)

            def _toggle_export(k: str, val: bool):
                ws.export_selection[k] = val

            ui.checkbox(
                f"{key} — {EXPORT_DESCRIPTIONS.get(key, key)}",
                value=ws.export_selection[key],
                on_change=lambda e, k=key: _toggle_export(k, bool(e.value)),
            )

        ui.select(["json", "csv"], value=ws.export_fmt, on_change=lambda e: setattr(ws, "export_fmt", str(e.value))).props(
            "label=Format outlined"
        )

        def do_export():
            sel = {k for k, v in ws.export_selection.items() if v}
            if not sel:
                set_status("Select at least one file.", "warning")
                body_content.refresh()
                return
            report = getattr(ws.instance, "generation_report", None) or {}
            if report.get("status") == "rejected":
                set_status(
                    "Exporting rejected instance — files will be written under a rejected/ folder.",
                    "warning",
                )
            out = export_instance(
                ws.instance,
                output_dir=f"benchmark_export_{ws.export_fmt}",
                fmt=ws.export_fmt,
                selection=sel,
            )
            set_status(f"Saved to {out}", "success")
            body_content.refresh()

        ui.button("Export selected files", on_click=do_export, icon="save").classes("bg-emerald-600 text-white mt-4").props(
            "unelevated no-caps"
        )

        with ui.row().classes("gap-3 mt-4"):
            ui.button("← Back to results", on_click=step_nav_handler(5)).props("outline no-caps")

            async def start_over():
                ws.reset_to_initial()
                await navigate_step(1)

            ui.button("Start over", on_click=start_over).props("outline no-caps color=negative")


def init_page():
    register_asset_static_routes()
    ui.page_title("EVRP Benchmark Generator")
    ui.add_head_html(
        """
        <style>
            body { background: linear-gradient(150deg, #ecfdf5 0%, #f8fafc 46%, #f1f5f9 100%) !important; }
            .q-card { border-radius: 14px !important; }
            .q-btn { border-radius: 10px !important; font-weight: 600 !important; letter-spacing: .01em; }
            .q-badge { border-radius: 8px !important; }
            .nicegui-content { width: 100%; }
            iframe { width: 100% !important; max-width: 100% !important; display: block !important; }
            .q-btn.q-btn--outline:not(.text-negative):not(.bg-negative) {
                color: #047857 !important;
                border-color: #6ee7b7 !important;
            }
        </style>
        """
    )
    with ui.column().style("width:60%; max-width:60%; margin:0 auto;").classes("p-4 md:p-8 gap-4"):
        step_header()
        body_content()


if __name__ in {"__main__", "__mp_main__"}:
    init_page()
    port = int(os.environ.get("PORT", "8080"))
    favicon_path = str((_APP_DIR / "assets" / "logo_app.png").resolve())
    ui.run(
        title="EVRP Benchmark Generator",
        host="0.0.0.0",
        port=port,
        favicon=favicon_path,
        reload=False,
    )

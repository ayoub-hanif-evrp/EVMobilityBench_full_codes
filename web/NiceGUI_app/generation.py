"""EVRP instance generation for the NiceGUI wizard."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evrp_instance_generator_framework import (
    EVFeatures,
    GenerationConfig,
    apply_customers_to_state,
    generate_instance,
    load_customers_from_csv,
)
from evrp_instance_generator_framework.variants import classic as classic_variant
from evrp_instance_generator_framework.variants import multi_depot as md_variant
from evrp_instance_generator_framework.variants import two_echelon as te_variant

try:
    from .state import WizardState
except ImportError:
    from state import WizardState  # noqa: I001 — script entry via web/NiceGUI_app/main.py


def build_generation_config(ws: WizardState) -> GenerationConfig:
    cp = ws.cust_params_dict()
    G = ws.movement_graph
    if G is None:
        raise RuntimeError("No movement graph")

    depot_nid = ws.depot_node_id
    if depot_nid is None:
        raise RuntimeError("Depot not snapped")

    depot_node_data = G.nodes[depot_nid]
    if ws.depot_facility_lat is not None and ws.depot_facility_lon is not None:
        depot_lat = float(ws.depot_facility_lat)
        depot_lon = float(ws.depot_facility_lon)
        depot_snap_m = ws.default_depot_snap_m()
    else:
        depot_lat = float(depot_node_data["y"])
        depot_lon = float(depot_node_data["x"])
        depot_snap_m = 10.0

    variant = cp.get("variant", "classic_evrptw")
    run_energy = ws.feasibility_scope == "time_and_energy"
    node_el = "srtm" if run_energy or cp.get("use_elevation") else "none"

    config_kw: dict[str, Any] = dict(
        city=ws.city,
        country=ws.country,
        depot_lat=depot_lat,
        depot_lon=depot_lon,
        depot_snap_max_dist_m=depot_snap_m,
        seed=cp.get("seed", 1234),
        num_stations=cp.get("num_stations", 5),
        energy_period=cp.get("energy_period", "off_peak"),
        demand_min=cp.get("demand_min", 5),
        demand_max=cp.get("demand_max", 20),
        node_elevation_provider=node_el,
        variant=variant,
        time_window_tightness=cp.get("time_window_tightness", "medium"),
    )

    if variant in ("classic_evrptw", "multi_depot_evrptw", "two_echelon_evrp"):
        config_kw["num_customers"] = cp.get("num_customers", 20)
        config_kw["num_clusters"] = cp.get("num_clusters", 3)
        config_kw["customer_pattern"] = cp.get("customer_pattern", "rc")

    if variant == "two_echelon_evrp":
        ns = int(cp.get("two_echelon_num_satellites", 3))
        locs = []
        for j in range(1, max(1, ns) + 1):
            if j < len(ws.md_depot_lat) and j < len(ws.md_depot_lon):
                locs.append((float(ws.md_depot_lat[j]), float(ws.md_depot_lon[j])))
            else:
                break
        if len(locs) >= 1:
            config_kw["satellite_locations"] = tuple(locs)
            config_kw["num_satellites"] = len(locs)
        else:
            config_kw["satellite_locations"] = ()
            config_kw["num_satellites"] = ns

    if variant == "multi_depot_evrptw":
        n_add = int(cp.get("num_additional_depots", 2))
        extras = ws.extra_depots or []
        if len(extras) >= n_add:
            config_kw["additional_depots"] = tuple((float(lat), float(lon)) for lat, lon in extras[:n_add])
        else:
            config_kw["num_additional_depots"] = n_add

    for k in (
        "cluster_max_radius_m",
        "cluster_min_separation_m",
        "customer_building_osm_min_candidates",
        "station_osm_min_candidates",
    ):
        if k in cp and cp[k] is not None:
            config_kw[k] = cp[k]

    return GenerationConfig(**config_kw)


def run_generation(ws: WizardState) -> tuple[Any | None, str | None]:
    cp = ws.cust_params_dict()
    variant = cp.get("variant", "classic_evrptw")
    ev = EVFeatures(**ws.ev_params_dict())
    G = ws.movement_graph
    if G is None:
        return None, "No road network loaded."

    run_energy = ws.feasibility_scope == "time_and_energy"
    config = build_generation_config(ws)

    try:
        csv_bytes = cp.get("customer_csv_bytes")
        if csv_bytes:
            imported_customers = load_customers_from_csv(csv_bytes)
            if variant == "classic_evrptw":
                state = classic_variant.prepare_graph_and_depot(config, ev, movement_graph=G)
                state = apply_customers_to_state(state, imported_customers)
                state = classic_variant.generate_stations(state)
                inst = classic_variant.finalize(
                    state,
                    compute_matrices=False,
                    run_energy_feasibility=run_energy,
                )
            elif variant == "multi_depot_evrptw":
                state = md_variant.prepare_graph_and_depots(config, ev, movement_graph=G)
                state = apply_customers_to_state(state, imported_customers)
                state = md_variant.generate_stations(state)
                inst = md_variant.finalize(
                    state,
                    compute_matrices=False,
                    run_energy_feasibility=run_energy,
                )
            else:
                state = te_variant.prepare_graph_and_depot(config, ev, movement_graph=G)
                state = te_variant.setup_satellites(state)
                state = apply_customers_to_state(state, imported_customers)
                state = te_variant.assign_customers_to_satellites(state)
                state = te_variant.generate_stations(state)
                inst = te_variant.finalize(
                    state,
                    compute_matrices=False,
                    run_energy_feasibility=run_energy,
                )
        else:
            inst = generate_instance(
                config=config,
                ev_features=ev,
                movement_graph=G,
                compute_matrices=False,
                run_energy_feasibility=run_energy,
            )
        return inst, None
    except Exception as exc:
        return None, str(exc)

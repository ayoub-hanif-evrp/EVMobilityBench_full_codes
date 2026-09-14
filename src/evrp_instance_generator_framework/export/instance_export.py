"""
Export a BenchmarkInstance to disk as structured files.

Supported formats: json, csv.

Exported files:
  metadata             – variant, city, seed, depot, EV settings, provenance counts.
  road_network_nodes   – every intersection (id, lat, lon, elevation_m).
  road_network_edges   – every directed edge (u→v, length, slope, speeds, travel times).
  customers            – customer records (demand, time windows, service time).
  stations             – station records (type, power, slots, green label, provenance).
  service_nodes        – ordered list linking matrix indices to road-network node IDs.
  feasibility          – feasibility check results.
  depots               – depot records (multi-depot variant).
  satellites           – satellite records (two-echelon variant).
"""

import csv
import json
import os
from dataclasses import asdict
from typing import Literal, Optional, Set

from ..service_graph.service_node_mapping import service_node_roles
from ..validation.instance_validator import is_instance_accepted

ALL_EXPORT_KEYS = [
    "metadata",
    "road_network_nodes",
    "road_network_edges",
    "customers",
    "stations",
    "service_nodes",
    "feasibility",
    "generation_report",
    "depots",
    "satellites",
]

EXPORT_DESCRIPTIONS = {
    "metadata":            "Instance metadata (variant, city, seed, depot, provenance counts)",
    "road_network_nodes":  "Road network nodes (id, lat, lon, elevation)",
    "road_network_edges":  "Road network edges (u→v, length, slope, speeds, travel times)",
    "customers":           "Customer records (demand, time windows, service time)",
    "stations":            "Station records (type, power, slots, green label, provenance)",
    "service_nodes":       "Service-node ordering (links to road-network node IDs)",
    "feasibility":         "Feasibility check results",
    "generation_report":   "Generation quality audit (counts, repairs, acceptance)",
    "depots":              "Depot records (multi-depot variant)",
    "satellites":          "Satellite records (two-echelon variant)",
}

# Keys common to every variant (road network, customers, stations, matrices, feasibility).
_BASE_EXPORT_KEYS = (
    "metadata",
    "road_network_nodes",
    "road_network_edges",
    "customers",
    "stations",
    "service_nodes",
    "feasibility",
)

# Variant-specific exports (only these are offered in UIs that filter by variant).
_VARIANT_EXTRA_EXPORT_KEYS: dict[str, tuple[str, ...]] = {
    "classic_evrptw": (),
    "multi_depot_evrptw": ("depots",),
    "two_echelon_evrp": ("satellites",),
}


def export_keys_for_variant(variant: str) -> tuple[str, ...]:
    """
    Return the export file keys that apply to ``variant``.

    Unknown variants fall back to ``ALL_EXPORT_KEYS`` so new variants still work.
    """
    if variant not in _VARIANT_EXTRA_EXPORT_KEYS:
        return tuple(ALL_EXPORT_KEYS)
    return _BASE_EXPORT_KEYS + _VARIANT_EXTRA_EXPORT_KEYS[variant]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _write_csv(path: str, rows: list, fieldnames: list) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_instance(
    instance,
    output_dir: str,
    fmt: Literal["json", "csv"] = "json",
    selection: Optional[Set[str]] = None,
) -> str:
    """
    Write selected instance data to ``output_dir``.

    Parameters
    ----------
    instance : BenchmarkInstance
    output_dir : str
    fmt : "json" or "csv"
    selection : set of keys to export, or None for all.

    Returns absolute path of output_dir.
    """
    accepted = is_instance_accepted(instance)
    if not accepted:
        base = os.path.basename(os.path.normpath(output_dir)) or "instance"
        parent = os.path.dirname(os.path.abspath(output_dir))
        output_dir = os.path.join(parent, "rejected", base)
    _ensure_dir(output_dir)
    ext = fmt
    want = selection if selection else set(ALL_EXPORT_KEYS)

    # ── metadata (variant-aware) ──────────────────────────────────────────
    if "metadata" in want:
        meta = {
            "variant": getattr(instance.metadata, "variant", "classic_evrptw"),
            "city": instance.metadata.city,
            "country": instance.metadata.country,
            "seed": instance.metadata.seed,
            "movement_node_count": instance.metadata.movement_node_count,
            "service_node_count": instance.metadata.service_node_count,
            "depot_facility_lat": float(instance.config.depot_lat),
            "depot_facility_lon": float(instance.config.depot_lon),
            "depot_node_id": instance.depot_node_id,
            "depot_snap_distance_m": float(
                instance.metadata.extra.get("depot_snap_distance_m", 0.0)
            ),
            "energy_period": instance.config.energy_period,
            "time_window_tightness": getattr(instance.metadata, "time_window_tightness", "medium"),
            "feasibility_level": getattr(instance.metadata, "feasibility_level", "validity_time_energy"),
            "num_customers": len(instance.customers),
            "num_stations": len(instance.stations),
            "depot_count": getattr(instance.metadata, "depot_count", 1),
            "satellite_count": getattr(instance.metadata, "satellite_count", 0),
            "station_count_observed_ev": getattr(instance.metadata, "station_count_observed_ev", 0),
            "station_count_proxy_host": getattr(instance.metadata, "station_count_proxy_host", 0),
            "station_count_synthetic": getattr(instance.metadata, "station_count_synthetic", 0),
            "elevation_enabled": getattr(instance.metadata, "elevation_enabled", True),
            "two_echelon_enabled": getattr(instance.metadata, "two_echelon_enabled", False),
        }
        if fmt == "json":
            _write_json(os.path.join(output_dir, f"metadata.{ext}"), meta)
        else:
            _write_csv(os.path.join(output_dir, f"metadata.{ext}"), [meta], list(meta.keys()))

    # ── road_network_nodes ────────────────────────────────────────────────
    if "road_network_nodes" in want:
        G = instance.movement_graph
        node_rows = [
            {
                "node_id": nid,
                "lat": float(d["y"]),
                "lon": float(d["x"]),
                "elevation_m": float(d.get("elevation_m", 0.0)),
            }
            for nid, d in G.nodes(data=True)
        ]
        if fmt == "json":
            _write_json(os.path.join(output_dir, f"road_network_nodes.{ext}"), node_rows)
        else:
            _write_csv(os.path.join(output_dir, f"road_network_nodes.{ext}"), node_rows,
                        ["node_id", "lat", "lon", "elevation_m"])

    # ── road_network_edges ────────────────────────────────────────────────
    if "road_network_edges" in want:
        G = instance.movement_graph
        edge_rows = [
            {
                "from_node": u, "to_node": v,
                "length_m": float(d.get("length_m", 0.0)),
                "slope_angle_rad": float(d.get("slope_angle_rad", 0.0)),
                "speed_limit_kph": float(d.get("speed_limit_kph", 0.0)),
                "free_flow_speed_kph": float(d.get("free_flow_speed_kph", 0.0)),
                "off_peak_travel_time_s": float(d.get("off_peak_travel_time_s", 0.0)),
                "midday_travel_time_s": float(d.get("midday_travel_time_s", 0.0)),
                "pm_peak_travel_time_s": float(d.get("pm_peak_travel_time_s", 0.0)),
            }
            for u, v, d in G.edges(data=True)
        ]
        edge_fields = [
            "from_node", "to_node", "length_m", "slope_angle_rad",
            "speed_limit_kph", "free_flow_speed_kph",
            "off_peak_travel_time_s", "midday_travel_time_s", "pm_peak_travel_time_s",
        ]
        if fmt == "json":
            _write_json(os.path.join(output_dir, f"road_network_edges.{ext}"), edge_rows)
        else:
            _write_csv(os.path.join(output_dir, f"road_network_edges.{ext}"), edge_rows, edge_fields)

    # ── customers ─────────────────────────────────────────────────────────
    if "customers" in want:
        cust_rows = [asdict(c) for c in instance.customers]
        if fmt == "json":
            _write_json(os.path.join(output_dir, f"customers.{ext}"), cust_rows)
        else:
            if cust_rows:
                _write_csv(os.path.join(output_dir, f"customers.{ext}"), cust_rows, list(cust_rows[0].keys()))

    # ── stations (with provenance) ────────────────────────────────────────
    if "stations" in want:
        stat_rows = []
        for s in instance.stations:
            d = asdict(s)
            d.pop("osm_tags", None)  # raw tags can be large; export separately if needed
            stat_rows.append(d)
        if fmt == "json":
            _write_json(os.path.join(output_dir, f"stations.{ext}"), stat_rows)
        else:
            if stat_rows:
                _write_csv(os.path.join(output_dir, f"stations.{ext}"), stat_rows, list(stat_rows[0].keys()))

    # ── service_nodes ─────────────────────────────────────────────────────
    if "service_nodes" in want:
        roles = service_node_roles(
            instance.depot_node_id, instance.customers, instance.stations
        )
        sn_rows = [
            {"index": i, "node_id": int(nid), "role": roles[i]}
            for i, nid in enumerate(instance.service_nodes)
        ]
        if fmt == "json":
            _write_json(os.path.join(output_dir, f"service_nodes.{ext}"), sn_rows)
        else:
            _write_csv(os.path.join(output_dir, f"service_nodes.{ext}"), sn_rows, ["index", "node_id", "role"])

    # ── feasibility ───────────────────────────────────────────────────────
    if "feasibility" in want:
        if fmt == "json":
            _write_json(os.path.join(output_dir, f"feasibility.{ext}"), instance.feasibility)
        else:
            flat = {k: str(v) for k, v in instance.feasibility.items()}
            _write_csv(os.path.join(output_dir, f"feasibility.{ext}"), [flat], list(flat.keys()))

    if "generation_report" in want and getattr(instance, "generation_report", None):
        if fmt == "json":
            _write_json(
                os.path.join(output_dir, f"generation_report.{ext}"),
                instance.generation_report,
            )
        else:
            flat = {k: str(v) for k, v in instance.generation_report.items()}
            _write_csv(
                os.path.join(output_dir, f"generation_report.{ext}"),
                [flat],
                list(flat.keys()),
            )

    # ── depots (multi-depot variant) ──────────────────────────────────────
    if "depots" in want and hasattr(instance, "depots") and instance.depots:
        depot_rows = [asdict(d) for d in instance.depots]
        if fmt == "json":
            _write_json(os.path.join(output_dir, f"depots.{ext}"), depot_rows)
        else:
            _write_csv(os.path.join(output_dir, f"depots.{ext}"), depot_rows, list(depot_rows[0].keys()))

    # ── satellites (two-echelon variant) ──────────────────────────────────
    if "satellites" in want and hasattr(instance, "satellites") and instance.satellites:
        sat_rows = [asdict(s) for s in instance.satellites]
        if fmt == "json":
            _write_json(os.path.join(output_dir, f"satellites.{ext}"), sat_rows)
        else:
            _write_csv(os.path.join(output_dir, f"satellites.{ext}"), sat_rows, list(sat_rows[0].keys()))

    return os.path.abspath(output_dir)

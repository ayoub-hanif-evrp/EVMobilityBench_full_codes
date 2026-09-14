"""Build generation_report.json audit payload (does not alter benchmark schema)."""

from __future__ import annotations

from statistics import mean, median
from typing import Any, Dict, List, Optional

from ..service_graph.service_node_mapping import service_node_role_counts
from ..types import BenchmarkInstance, GenerationConfig, GenerationRepairSummary


def _snap_summary(customers, stations, depot_snap_m: float) -> Dict[str, float]:
    c_snaps = [float(c.snap_distance_m) for c in customers]
    s_snaps = [float(s.snap_distance_m) for s in stations]
    out: Dict[str, float] = {"depot_m": float(depot_snap_m)}
    if c_snaps:
        out["customer_mean_m"] = float(mean(c_snaps))
        out["customer_max_m"] = float(max(c_snaps))
    else:
        out["customer_mean_m"] = 0.0
        out["customer_max_m"] = 0.0
    if s_snaps:
        out["station_mean_m"] = float(mean(s_snaps))
        out["station_max_m"] = float(max(s_snaps))
    else:
        out["station_mean_m"] = 0.0
        out["station_max_m"] = 0.0
    return out


def _time_window_summary(customers) -> Dict[str, float]:
    if not customers:
        return {
            "min_width_s": 0.0,
            "median_width_s": 0.0,
            "mean_width_s": 0.0,
            "max_width_s": 0.0,
            "mean_service_time_s": 0.0,
            "max_service_to_window_ratio": 0.0,
        }
    widths = [float(c.time_close_s - c.time_open_s) for c in customers]
    services = [float(c.service_time_s) for c in customers]
    ratios = [
        (float(c.service_time_s) + float(c.parking_time_s)) / max(1.0, float(c.time_close_s - c.time_open_s))
        for c in customers
    ]
    return {
        "min_width_s": float(min(widths)),
        "median_width_s": float(median(widths)),
        "mean_width_s": float(mean(widths)),
        "max_width_s": float(max(widths)),
        "mean_service_time_s": float(mean(services)),
        "max_service_to_window_ratio": float(max(ratios)),
    }


def _station_source_summary(stations) -> Dict[str, int]:
    counts = {"observed_ev": 0, "proxy_host": 0, "synthetic": 0}
    for s in stations:
        st = getattr(s, "station_source_type", "synthetic")
        if st == "observed_ev":
            counts["observed_ev"] += 1
        elif st == "proxy_host":
            counts["proxy_host"] += 1
        else:
            counts["synthetic"] += 1
    return counts


def _overlap_counts(instance: BenchmarkInstance) -> tuple[int, int]:
    customer_ids = {int(c.movement_node_id) for c in instance.customers}
    station_ids = {int(s.movement_node_id) for s in instance.stations}
    overlap = len(customer_ids & station_ids)
    dup = len(instance.service_nodes) - len(set(instance.service_nodes))
    return dup, overlap


def build_generation_report(
    instance: BenchmarkInstance,
    config: GenerationConfig,
    repair_summary: Optional[GenerationRepairSummary] = None,
    *,
    accepted: bool = True,
    failure_reasons: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Structured generation audit report (optional export file)."""
    rs = repair_summary or GenerationRepairSummary()
    role_counts = service_node_role_counts(
        instance.depot_node_id, instance.customers, instance.stations
    )
    dup, overlap = _overlap_counts(instance)
    depot_snap = float(instance.metadata.extra.get("depot_snap_distance_m", 0.0))

    return {
        "status": "accepted" if accepted else "rejected",
        "failure_reasons": list(failure_reasons or []),
        "city": config.city,
        "country": config.country,
        "variant": getattr(instance.metadata, "variant", config.variant),
        "seed": int(config.seed),
        "customer_pattern": config.customer_pattern,
        "time_window_tightness": config.time_window_tightness,
        "num_customers_requested": int(config.num_customers),
        "num_customers_exported": len(instance.customers),
        "num_stations_requested": int(config.num_stations),
        "num_stations_exported": len(instance.stations),
        "service_node_count": len(instance.service_nodes),
        "role_counts": role_counts,
        "duplicate_service_node_count": dup,
        "customer_station_overlap_count": overlap,
        "snap_distance_summary": _snap_summary(instance.customers, instance.stations, depot_snap),
        "time_window_summary": _time_window_summary(instance.customers),
        "station_source_summary": _station_source_summary(instance.stations),
        "road_graph_summary": {
            "movement_node_count": int(instance.metadata.movement_node_count),
            "edge_count": int(instance.movement_graph.number_of_edges())
            if instance.movement_graph is not None
            else 0,
            "service_nodes_reachable": bool(
                instance.feasibility.get("service_graph", {}).get("ok", True)
            ),
        },
        "repair_summary": {
            "customer_resamples": int(rs.customer_resamples),
            "station_resamples": int(rs.station_resamples),
            "time_window_repairs": int(rs.time_window_repairs),
            "duplicate_rejections": int(rs.duplicate_rejections),
            "tight_window_warnings": int(rs.tight_window_warnings),
            "customer_rejection_reasons": dict(rs.customer_rejection_reasons),
        },
    }

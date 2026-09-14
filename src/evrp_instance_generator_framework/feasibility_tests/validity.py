"""Structural validity checks (demands, TW consistency, graph membership, two-echelon loads)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..types import CustomerRecord, DepotRecord, SatelliteRecord, StationRecord


def check_validity(
    *,
    movement_graph: Any,
    customers: List[CustomerRecord],
    stations: List[StationRecord],
    depots: Optional[List[DepotRecord]] = None,
    satellites: Optional[List[SatelliteRecord]] = None,
    depot_node_id: Optional[int] = None,
    service_nodes: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Return ``{"ok": bool, "issues": [...], "errors": [...], "warnings": [...]}``."""
    issues: List[Dict[str, Any]] = []
    errors: List[str] = []
    warnings: List[str] = []

    if depots:
        for d in depots:
            if int(d.time_open_s) > int(d.time_close_s):
                issues.append(
                    {
                        "type": "depot_time_window",
                        "depot_id": d.id,
                        "note": "time_open_s exceeds time_close_s",
                    }
                )

    for c in customers:
        if float(c.demand) < 0:
            issues.append({"type": "negative_demand", "customer_id": c.id})
        if int(c.time_open_s) > int(c.time_close_s):
            issues.append({"type": "customer_time_window", "customer_id": c.id})
        nid = int(c.movement_node_id)
        if movement_graph is not None and nid not in movement_graph:
            issues.append(
                {"type": "customer_node_not_in_graph", "customer_id": c.id, "movement_node_id": nid}
            )

    for s in stations:
        if int(s.time_open_s) > int(s.time_close_s):
            issues.append({"type": "station_time_window", "station_id": s.id})

    if satellites:
        by_id = {c.id: c for c in customers}
        for sat in satellites:
            if int(sat.time_open_s) > int(sat.time_close_s):
                issues.append({"type": "satellite_time_window", "satellite_id": sat.id})
            nid = int(sat.movement_node_id)
            if movement_graph is not None and nid not in movement_graph:
                issues.append(
                    {"type": "satellite_node_not_in_graph", "satellite_id": sat.id, "movement_node_id": nid}
                )
            load = sum(by_id[i].demand for i in sat.assigned_customer_ids if i in by_id)
            if load > int(sat.capacity):
                issues.append(
                    {
                        "type": "satellite_load_exceeds_capacity",
                        "satellite_id": sat.id,
                        "assigned_load": load,
                        "capacity": int(sat.capacity),
                    }
                )

    customer_node_ids = {int(c.movement_node_id) for c in customers}
    station_node_ids = {int(s.movement_node_id) for s in stations}
    overlap = customer_node_ids & station_node_ids
    if overlap:
        msg = f"customer_station_node_overlap: {sorted(overlap)[:5]}"
        issues.append({"type": "customer_station_overlap", "nodes": sorted(overlap)})
        errors.append(msg)

    if depot_node_id is not None:
        dep = int(depot_node_id)
        if dep in customer_node_ids:
            issues.append({"type": "depot_customer_overlap", "movement_node_id": dep})
            errors.append(f"depot node {dep} overlaps a customer")
        if dep in station_node_ids:
            issues.append({"type": "depot_station_overlap", "movement_node_id": dep})
            errors.append(f"depot node {dep} overlaps a station")

    if len(customer_node_ids) != len(customers):
        issues.append({"type": "duplicate_customer_nodes"})
        errors.append("duplicate customer movement_node_id values")

    if len(station_node_ids) != len(stations):
        issues.append({"type": "duplicate_station_nodes"})
        errors.append("duplicate station movement_node_id values")

    if service_nodes is not None and len(service_nodes) != len(set(service_nodes)):
        issues.append({"type": "duplicate_service_nodes"})
        errors.append("duplicate entries in service_nodes ordering")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "errors": errors,
        "warnings": warnings,
    }

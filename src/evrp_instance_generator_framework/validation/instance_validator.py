"""Post-generation validation and acceptance checks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from ..exceptions import EvrpValidationError
from ..service_graph.service_node_mapping import (
    service_node_role_counts,
    validate_service_nodes,
)
from ..types import BenchmarkInstance, GenerationRepairSummary


def _service_graph_diagnostics(instance: BenchmarkInstance) -> Dict[str, Any]:
    role_counts = service_node_role_counts(
        instance.depot_node_id, instance.customers, instance.stations
    )
    customer_ids = {int(c.movement_node_id) for c in instance.customers}
    station_ids = {int(s.movement_node_id) for s in instance.stations}
    dup = len(instance.service_nodes) - len(set(instance.service_nodes))
    overlap = len(customer_ids & station_ids)
    ok = (
        dup == 0
        and overlap == 0
        and role_counts["depot"] == 1
        and role_counts["customer"] == len(instance.customers)
        and role_counts["station"] == len(instance.stations)
        and len(instance.service_nodes)
        == 1 + len(instance.customers) + len(instance.stations)
    )
    errors: List[str] = []
    if dup:
        errors.append(f"duplicate_service_node_count={dup}")
    if overlap:
        errors.append(f"customer_station_overlap_count={overlap}")
    if role_counts["customer"] != len(instance.customers):
        errors.append("customer role count mismatch")
    if role_counts["station"] != len(instance.stations):
        errors.append("station role count mismatch")
    return {
        "ok": ok,
        "role_counts": role_counts,
        "duplicate_service_node_count": dup,
        "customer_station_overlap_count": overlap,
        "errors": errors,
        "warnings": [],
    }


def _graph_membership_checks(instance: BenchmarkInstance) -> List[str]:
    errors: List[str] = []
    G = instance.movement_graph
    if G is None:
        return errors
    node_set: Set[int] = set(int(n) for n in G.nodes)
    for nid in instance.service_nodes:
        if int(nid) not in node_set:
            errors.append(f"service node {nid} not in road graph")
    for u, v, _ in G.edges(data=True):
        if int(u) not in node_set or int(v) not in node_set:
            errors.append(f"edge ({u},{v}) references missing node")
            break
    return errors


def validate_benchmark_instance(instance: BenchmarkInstance) -> None:
    """Raise ``EvrpValidationError`` if service-graph contract is violated."""
    validate_service_nodes(
        instance.service_nodes,
        depot_node_id=instance.depot_node_id,
        customers=instance.customers,
        stations=instance.stations,
    )
    if len(instance.stations) != int(instance.config.num_stations):
        raise EvrpValidationError(
            f"Station count {len(instance.stations)} != requested {instance.config.num_stations}."
        )
    if len(instance.customers) != int(instance.config.num_customers):
        raise EvrpValidationError(
            f"Customer count {len(instance.customers)} != requested {instance.config.num_customers}."
        )
    graph_errors = _graph_membership_checks(instance)
    if graph_errors:
        raise EvrpValidationError("; ".join(graph_errors[:5]))


def enrich_feasibility_report(
    feasibility: Dict[str, Any],
    instance: BenchmarkInstance,
) -> Dict[str, Any]:
    """Attach ``service_graph`` section and recompute ``all_passed`` honestly."""
    out = dict(feasibility)
    sg = _service_graph_diagnostics(instance)
    out["service_graph"] = sg

    graph_errors = _graph_membership_checks(instance)
    if graph_errors:
        validity = dict(out.get("validity") or {"ok": True, "issues": []})
        issues = list(validity.get("issues") or [])
        for msg in graph_errors:
            issues.append({"type": "graph_membership", "note": msg})
        validity["ok"] = False
        validity["issues"] = issues
        validity.setdefault("errors", [])
        validity["errors"] = list(validity.get("errors", [])) + graph_errors
        out["validity"] = validity

    if not sg.get("ok"):
        validity = dict(out.get("validity") or {"ok": True, "issues": []})
        validity["ok"] = False
        for err in sg.get("errors", []):
            validity.setdefault("issues", []).append({"type": "service_graph", "note": err})
        out["validity"] = validity

    en_ok = out.get("energy_feasibility", {}).get("skipped") or out.get("energy_feasibility", {}).get("ok", True)
    out["all_passed"] = bool(
        out.get("validity", {}).get("ok")
        and out.get("time_feasibility", {}).get("ok")
        and en_ok
        and sg.get("ok")
    )
    return out


def is_instance_accepted(instance: BenchmarkInstance) -> bool:
    """Acceptance per paper export criteria."""
    meta = instance.metadata
    n_c = len(instance.customers)
    n_s = len(instance.stations)
    if meta.service_node_count != 1 + n_c + n_s:
        return False
    sg = instance.feasibility.get("service_graph") or _service_graph_diagnostics(instance)
    if sg.get("duplicate_service_node_count", 0) != 0:
        return False
    if sg.get("customer_station_overlap_count", 0) != 0:
        return False
    rc = sg.get("role_counts") or {}
    if rc.get("depot") != 1 or rc.get("customer") != n_c or rc.get("station") != n_s:
        return False
    if not instance.feasibility.get("validity", {}).get("ok", False):
        return False
    if not instance.feasibility.get("time_feasibility", {}).get("ok", False):
        return False
    en = instance.feasibility.get("energy_feasibility", {})
    if not en.get("skipped") and not en.get("ok", False):
        return False
    return bool(instance.feasibility.get("all_passed", False))


def attach_post_finalize_artifacts(
    instance: BenchmarkInstance,
    repair_summary: Optional[GenerationRepairSummary] = None,
) -> BenchmarkInstance:
    """Validate, enrich feasibility, and build generation report."""
    from .generation_report import build_generation_report

    validate_benchmark_instance(instance)
    instance.feasibility = enrich_feasibility_report(instance.feasibility, instance)
    accepted = is_instance_accepted(instance)
    reasons: List[str] = []
    if not accepted:
        sg = instance.feasibility.get("service_graph", {})
        reasons.extend(sg.get("errors") or [])
        if not instance.feasibility.get("validity", {}).get("ok"):
            reasons.append("validity check failed")
        if not instance.feasibility.get("time_feasibility", {}).get("ok"):
            reasons.append("time feasibility check failed")
        en = instance.feasibility.get("energy_feasibility", {})
        if not en.get("skipped") and not en.get("ok", False):
            reasons.append("energy feasibility check failed")

    instance.generation_report = build_generation_report(
        instance,
        instance.config,
        repair_summary,
        accepted=accepted,
        failure_reasons=reasons,
    )
    instance.metadata.extra["acceptance_status"] = (
        "accepted" if accepted else "rejected"
    )
    return instance

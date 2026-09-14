"""Energy screening: direct leg or depot→station→customer within battery."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..service_graph.energy_consumption import compute_energy_matrix
from ..types import CustomerRecord, DepotRecord, EVFeatures, GenerationConfig, StationRecord


def _depot_energy_matrices(
    movement_graph: Any,
    depots: List[DepotRecord],
    customers: List[CustomerRecord],
    stations: List[StationRecord],
    period: Any,
    ev_features: EVFeatures,
    travel_time_precomputed: Optional[np.ndarray],
) -> List[np.ndarray]:
    mats: List[np.ndarray] = []
    for d in depots:
        service_nodes = [int(d.movement_node_id)]
        service_nodes.extend(int(c.movement_node_id) for c in customers)
        service_nodes.extend(int(s.movement_node_id) for s in stations)
        mats.append(
            compute_energy_matrix(
                movement_graph,
                service_nodes,
                period=period,
                ev_features=ev_features,
                precomputed_travel_time=travel_time_precomputed,
            )
        )
    return mats


def check_energy_classic(
    config: GenerationConfig,
    ev_features: EVFeatures,
    customers: List[CustomerRecord],
    stations: List[StationRecord],
    energy_matrix_kwh: Optional[np.ndarray],
    *,
    one_hop_via_station: bool,
) -> Dict[str, Any]:
    """If ``energy_matrix_kwh`` is missing, returns skipped=True (not a failure)."""
    if energy_matrix_kwh is None:
        return {
            "ok": True,
            "skipped": True,
            "reason": "no_energy_matrix",
            "issues": [],
        }

    cap = float(ev_features.battery_capacity_kwh)
    n_cust = len(customers)
    n_st = len(stations)
    depot_idx = 0
    cust_start = 1
    st_start = 1 + n_cust
    issues: List[Dict[str, Any]] = []
    ok = True

    if not one_hop_via_station:
        for i, c in enumerate(customers):
            ci = cust_start + i
            if float(energy_matrix_kwh[depot_idx, ci]) > cap:
                ok = False
                issues.append({"type": "energy_direct_leg", "customer_id": c.id})
        return {
            "ok": ok,
            "skipped": False,
            "issues": issues,
            "battery_capacity_kwh": cap,
            "mode": "direct_only",
        }

    for i, c in enumerate(customers):
        ci = cust_start + i
        if float(energy_matrix_kwh[depot_idx, ci]) <= cap:
            continue
        found = False
        for j in range(n_st):
            si = st_start + j
            if float(energy_matrix_kwh[depot_idx, si]) <= cap and float(energy_matrix_kwh[si, ci]) <= cap:
                found = True
                break
        if not found:
            ok = False
            issues.append({"type": "energy_support", "customer_id": c.id})

    return {
        "ok": ok,
        "skipped": False,
        "issues": issues,
        "battery_capacity_kwh": cap,
        "mode": "direct_or_one_station_hop",
    }


def check_energy_multi_depot(
    config: GenerationConfig,
    ev_features: EVFeatures,
    movement_graph: Any,
    depots: List[DepotRecord],
    customers: List[CustomerRecord],
    stations: List[StationRecord],
    energy_matrix_kwh: Optional[np.ndarray],
    period: Any,
    *,
    one_hop_via_station: bool,
) -> Dict[str, Any]:
    """Per customer: some depot admits direct or one-stop charging within battery."""
    if energy_matrix_kwh is None:
        return {"ok": True, "skipped": True, "reason": "no_energy_matrix", "issues": []}

    if not one_hop_via_station:
        return check_energy_classic(
            config,
            ev_features,
            customers,
            stations,
            energy_matrix_kwh,
            one_hop_via_station=False,
        )

    cap = float(ev_features.battery_capacity_kwh)
    n_cust = len(customers)
    n_st = len(stations)

    energy_mats: List[np.ndarray]
    if len(depots) == 1:
        energy_mats = [energy_matrix_kwh]
    else:
        energy_mats = _depot_energy_matrices(
            movement_graph, depots, customers, stations, period, ev_features, None
        )

    issues: List[Dict[str, Any]] = []
    ok = True

    for i, c in enumerate(customers):
        ci = 1 + i
        ok_any = False
        for E in energy_mats:
            if float(E[0, ci]) <= cap:
                ok_any = True
                break
            for j in range(n_st):
                si = 1 + n_cust + j
                if float(E[0, si]) <= cap and float(E[si, ci]) <= cap:
                    ok_any = True
                    break
            if ok_any:
                break
        if not ok_any:
            ok = False
            issues.append(
                {
                    "type": "energy_support",
                    "customer_id": c.id,
                    "note": "No depot admits direct or one-stop charging within battery.",
                }
            )

    return {
        "ok": ok,
        "skipped": False,
        "issues": issues,
        "battery_capacity_kwh": cap,
        "mode": "multi_depot_direct_or_one_hop",
    }

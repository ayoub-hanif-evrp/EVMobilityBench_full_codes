"""Time-window screening from depot(s)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..types import CustomerRecord, DepotRecord, EVFeatures, GenerationConfig, StationRecord


def check_time_classic(
    config: GenerationConfig,
    ev_features: EVFeatures,
    customers: List[CustomerRecord],
    stations: List[StationRecord],
    travel_time_matrix_s: Optional[np.ndarray],
    depot_to_node_time: Optional[Dict[int, float]],
) -> Dict[str, Any]:
    """
    Per customer: arrival at depot open + travel from depot; service must finish inside
    customer TW. If a full pairwise matrix exists, also require return to depot by depot close.
    """
    issues: List[Dict[str, Any]] = []
    sm = float(ev_features.speed_multiplier)
    depot_idx = 0
    cust_start = 1
    n_cust = len(customers)

    ok = True
    for i, c in enumerate(customers):
        ci = cust_start + i
        if travel_time_matrix_s is not None:
            arrival_s = float(config.depot_time_open_s + travel_time_matrix_s[depot_idx, ci])
            earliest_start_s = max(arrival_s, float(c.time_open_s))
            latest_finish_s = earliest_start_s + float(c.service_time_s)
            if latest_finish_s > float(c.time_close_s):
                ok = False
                issues.append({"type": "time_window", "customer_id": c.id})
            tt_back = float(travel_time_matrix_s[ci, depot_idx])
            if latest_finish_s + tt_back > float(config.depot_time_close_s):
                ok = False
                issues.append({"type": "return_to_depot_deadline", "customer_id": c.id})
        elif depot_to_node_time is not None:
            raw_tt = depot_to_node_time.get(c.movement_node_id)
            if raw_tt is None:
                ok = False
                issues.append({"type": "unreachable_customer", "customer_id": c.id})
                continue
            arrival_s = float(config.depot_time_open_s) + float(raw_tt) / sm
            earliest_start_s = max(arrival_s, float(c.time_open_s))
            latest_finish_s = earliest_start_s + float(c.service_time_s)
            if latest_finish_s > float(c.time_close_s):
                ok = False
                issues.append({"type": "time_window", "customer_id": c.id})

    return {
        "ok": ok,
        "issues": issues,
        "mode": "matrix" if travel_time_matrix_s is not None else "depot_times_only",
        "num_customers": n_cust,
        "num_stations": len(stations),
    }


def check_time_multi_depot(
    config: GenerationConfig,
    ev_features: EVFeatures,
    depots: List[DepotRecord],
    depot_forward: List[Dict[int, float]],
    depot_return: List[Dict[int, float]],
    customers: List[CustomerRecord],
) -> Dict[str, Any]:
    """A customer passes if at least one depot can serve forward + service + return within hours."""
    issues: List[Dict[str, Any]] = []
    sm = float(ev_features.speed_multiplier)

    ok = True
    for c in customers:
        tw_ok = False
        for d_idx, drec in enumerate(depots):
            fwd = depot_forward[d_idx].get(c.movement_node_id)
            ret = depot_return[d_idx].get(c.movement_node_id)
            if fwd is None or ret is None:
                continue
            fwd_s = float(fwd) / sm
            ret_s = float(ret) / sm
            arrival = float(drec.time_open_s) + fwd_s
            earliest_start = max(arrival, float(c.time_open_s))
            latest_finish = earliest_start + float(c.service_time_s)
            if latest_finish > float(c.time_close_s):
                continue
            if latest_finish + ret_s > float(drec.time_close_s):
                continue
            tw_ok = True
            break
        if not tw_ok:
            ok = False
            issues.append(
                {
                    "type": "time_window_no_depot",
                    "customer_id": c.id,
                    "note": "No depot can serve this customer within TW and depot closing times.",
                }
            )

    return {"ok": ok, "issues": issues, "num_customers": len(customers), "num_depots": len(depots)}


def check_satellites_reachable(
    satellites: List[Any],
    depot_to_node_time: Optional[Dict[int, float]],
) -> Dict[str, Any]:
    """Satellite hubs must be reachable from the primary depot when times are known."""
    if depot_to_node_time is None:
        return {"ok": True, "issues": [], "skipped": True}
    issues: List[Dict[str, Any]] = []
    ok = True
    for s in satellites:
        if depot_to_node_time.get(s.movement_node_id) is None:
            ok = False
            issues.append({"type": "satellite_unreachable", "satellite_id": s.id})
    return {"ok": ok, "issues": issues, "skipped": False}

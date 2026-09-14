"""Assemble validity, time, and energy test results for each variant."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..types import (
    CustomerRecord,
    DepotRecord,
    EVFeatures,
    GenerationConfig,
    SatelliteRecord,
    StationRecord,
)
from .energy_feasibility import check_energy_classic, check_energy_multi_depot
from .time_feasibility import check_satellites_reachable, check_time_classic, check_time_multi_depot
from .validity import check_validity

SCHEMA_VERSION = "feasibility_tests_v1"
REPORT_MODE = "validity_time_energy"


def _all_passed(
    validity: Dict[str, Any],
    time_f: Dict[str, Any],
    energy_f: Dict[str, Any],
) -> bool:
    en_ok = energy_f.get("skipped") or energy_f.get("ok", True)
    return bool(validity.get("ok")) and bool(time_f.get("ok")) and bool(en_ok)


def build_classic_report(
    *,
    movement_graph: Any,
    config: GenerationConfig,
    ev_features: EVFeatures,
    depot_node_id: int,
    customers: List[CustomerRecord],
    stations: List[StationRecord],
    travel_time_matrix_s: Optional[np.ndarray],
    energy_matrix_kwh: Optional[np.ndarray],
    depot_to_node_time: Optional[Dict[int, float]],
    compute_matrices: bool,
    run_energy_feasibility: bool,
    service_nodes: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Classic single-depot: three tests whenever data allows."""
    validity = check_validity(
        movement_graph=movement_graph,
        customers=customers,
        stations=stations,
        depots=None,
        satellites=None,
        depot_node_id=depot_node_id,
        service_nodes=service_nodes,
    )

    tt = travel_time_matrix_s if (compute_matrices or run_energy_feasibility) else None
    time_f = check_time_classic(
        config,
        ev_features,
        customers,
        stations,
        tt,
        depot_to_node_time,
    )

    energy_matrix = energy_matrix_kwh if (compute_matrices or run_energy_feasibility) else None
    energy_f = check_energy_classic(
        config,
        ev_features,
        customers,
        stations,
        energy_matrix,
        one_hop_via_station=config.feasibility_max_customers_one_hop_via_station,
    )

    note = None
    if run_energy_feasibility and not compute_matrices:
        note = (
            f"Feasibility ({REPORT_MODE}) evaluated with period={config.energy_period!r}; "
            "matrices not stored on instance (compute_matrices=False)."
        )

    out: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "feasibility_report_mode": REPORT_MODE,
        "feasibility_variant": "classic_evrptw",
        "validity": validity,
        "time_feasibility": time_f,
        "energy_feasibility": energy_f,
        "all_passed": _all_passed(validity, time_f, energy_f),
        "depot_node_id": int(depot_node_id),
        "time_window_tightness": config.time_window_tightness,
    }
    if note:
        out["note"] = note
    return out


def build_multi_depot_report(
    *,
    movement_graph: Any,
    config: GenerationConfig,
    ev_features: EVFeatures,
    depot_node_id: int,
    depots: List[DepotRecord],
    depot_forward: List[Dict[int, float]],
    depot_return: List[Dict[int, float]],
    customers: List[CustomerRecord],
    stations: List[StationRecord],
    travel_time_matrix_s: Optional[np.ndarray],
    energy_matrix_kwh: Optional[np.ndarray],
    compute_matrices: bool,
    run_energy_feasibility: bool,
    service_nodes: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Multi-depot time screening uses per-depot forward/return maps."""
    _ = travel_time_matrix_s  # retained for API symmetry; time test uses depot dicts

    validity = check_validity(
        movement_graph=movement_graph,
        customers=customers,
        stations=stations,
        depots=depots,
        satellites=None,
        depot_node_id=depot_node_id,
        service_nodes=service_nodes,
    )

    time_f = check_time_multi_depot(
        config, ev_features, depots, depot_forward, depot_return, customers
    )

    energy_matrix = energy_matrix_kwh if (compute_matrices or run_energy_feasibility) else None
    energy_f = check_energy_multi_depot(
        config,
        ev_features,
        movement_graph,
        depots,
        customers,
        stations,
        energy_matrix,
        config.energy_period,
        one_hop_via_station=config.feasibility_max_customers_one_hop_via_station,
    )

    note = None
    if run_energy_feasibility and not compute_matrices:
        note = (
            f"Feasibility ({REPORT_MODE}) evaluated with period={config.energy_period!r}; "
            "matrices not stored on instance (compute_matrices=False)."
        )

    out: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "feasibility_report_mode": REPORT_MODE,
        "feasibility_variant": "multi_depot_evrptw",
        "validity": validity,
        "time_feasibility": time_f,
        "energy_feasibility": energy_f,
        "all_passed": _all_passed(validity, time_f, energy_f),
        "depot_node_id": int(depot_node_id),
        "depot_count": len(depots),
        "time_window_tightness": config.time_window_tightness,
        "multi_depot": True,
    }
    if note:
        out["note"] = note
    return out


def build_two_echelon_report(
    *,
    movement_graph: Any,
    config: GenerationConfig,
    ev_features: EVFeatures,
    depot_node_id: int,
    customers: List[CustomerRecord],
    stations: List[StationRecord],
    satellites: List[SatelliteRecord],
    travel_time_matrix_s: Optional[np.ndarray],
    energy_matrix_kwh: Optional[np.ndarray],
    depot_to_node_time: Optional[Dict[int, float]],
    compute_matrices: bool,
    run_energy_feasibility: bool,
    service_nodes: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Same three tests as classic, plus satellite structural checks and depot reachability."""
    validity = check_validity(
        movement_graph=movement_graph,
        customers=customers,
        stations=stations,
        depots=None,
        satellites=satellites,
        depot_node_id=depot_node_id,
        service_nodes=service_nodes,
    )

    tt = travel_time_matrix_s if (compute_matrices or run_energy_feasibility) else None
    time_f = check_time_classic(
        config,
        ev_features,
        customers,
        stations,
        tt,
        depot_to_node_time,
    )

    energy_matrix = energy_matrix_kwh if (compute_matrices or run_energy_feasibility) else None
    energy_f = check_energy_classic(
        config,
        ev_features,
        customers,
        stations,
        energy_matrix,
        one_hop_via_station=config.feasibility_max_customers_one_hop_via_station,
    )

    sat_r = check_satellites_reachable(satellites, depot_to_node_time)

    note = None
    if run_energy_feasibility and not compute_matrices:
        note = (
            f"Feasibility ({REPORT_MODE}) evaluated with period={config.energy_period!r}; "
            "matrices not stored on instance (compute_matrices=False)."
        )

    all_ok = _all_passed(validity, time_f, energy_f)
    if not sat_r.get("skipped") and not sat_r.get("ok", True):
        all_ok = False

    out: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "feasibility_report_mode": REPORT_MODE,
        "feasibility_variant": "two_echelon_evrp",
        "validity": validity,
        "time_feasibility": time_f,
        "energy_feasibility": energy_f,
        "satellite_reachability": sat_r,
        "all_passed": all_ok,
        "depot_node_id": int(depot_node_id),
        "time_window_tightness": config.time_window_tightness,
        "two_echelon": True,
        "satellite_count": len(satellites),
    }
    if note:
        out["note"] = note
    return out

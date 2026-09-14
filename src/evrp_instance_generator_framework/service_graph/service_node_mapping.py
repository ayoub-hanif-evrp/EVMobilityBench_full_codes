from __future__ import annotations

from typing import Dict, List, Sequence

from ..exceptions import EvrpValidationError
from ..types import CustomerRecord, StationRecord


def build_service_nodes(
    depot_node_id: int,
    customers: List[CustomerRecord],
    stations: List[StationRecord],
) -> List[int]:
    """
    Create the ordered list of service nodes used as the row/column order
    for all matrices:
      index 0: depot
      index 1..|C|: customers
      index 1+|C|..: stations
    """
    service_nodes: List[int] = [int(depot_node_id)]
    service_nodes.extend(int(c.movement_node_id) for c in customers)
    service_nodes.extend(int(s.movement_node_id) for s in stations)
    validate_service_nodes(
        service_nodes,
        depot_node_id=int(depot_node_id),
        customers=customers,
        stations=stations,
    )
    return service_nodes


def service_node_roles(
    depot_node_id: int,
    customers: Sequence[CustomerRecord],
    stations: Sequence[StationRecord],
) -> List[str]:
    """Positional roles matching matrix index order (not graph-node membership)."""
    roles = ["depot"]
    roles.extend("customer" for _ in customers)
    roles.extend("station" for _ in stations)
    return roles


def service_node_role_counts(
    depot_node_id: int,
    customers: Sequence[CustomerRecord],
    stations: Sequence[StationRecord],
) -> Dict[str, int]:
    roles = service_node_roles(depot_node_id, customers, stations)
    return {
        "depot": roles.count("depot"),
        "customer": roles.count("customer"),
        "station": roles.count("station"),
    }


def validate_service_nodes(
    service_nodes: Sequence[int],
    *,
    depot_node_id: int,
    customers: Sequence[CustomerRecord],
    stations: Sequence[StationRecord],
) -> None:
    """
    Strict service-graph integrity checks (paper ordering contract).

    Raises ``EvrpValidationError`` when the ordered service layer is invalid.
    """
    n_c = len(customers)
    n_s = len(stations)
    expected_len = 1 + n_c + n_s
    if len(service_nodes) != expected_len:
        raise EvrpValidationError(
            f"service_nodes length {len(service_nodes)} != 1 + customers + stations "
            f"({expected_len})."
        )

    roles = service_node_roles(depot_node_id, customers, stations)
    role_counts = {
        "depot": roles.count("depot"),
        "customer": roles.count("customer"),
        "station": roles.count("station"),
    }
    if role_counts["depot"] != 1:
        raise EvrpValidationError(f"Expected exactly one depot role, got {role_counts['depot']}.")
    if role_counts["customer"] != n_c:
        raise EvrpValidationError(
            f"Expected {n_c} customer roles, got {role_counts['customer']}."
        )
    if role_counts["station"] != n_s:
        raise EvrpValidationError(
            f"Expected {n_s} station roles, got {role_counts['station']}."
        )

    if service_nodes[0] != int(depot_node_id):
        raise EvrpValidationError(
            f"service_nodes[0] must be depot node {depot_node_id}, got {service_nodes[0]}."
        )

    for i in range(1, 1 + n_c):
        if roles[i] != "customer":
            raise EvrpValidationError(
                f"service_nodes[{i}] must have role 'customer', got {roles[i]!r}."
            )

    for i in range(1 + n_c, 1 + n_c + n_s):
        if roles[i] != "station":
            raise EvrpValidationError(
                f"service_nodes[{i}] must have role 'station', got {roles[i]!r}."
            )

    customer_node_ids = {int(c.movement_node_id) for c in customers}
    station_node_ids = {int(s.movement_node_id) for s in stations}
    depot_nid = int(depot_node_id)

    if not customer_node_ids.isdisjoint(station_node_ids):
        overlap = sorted(customer_node_ids & station_node_ids)
        raise EvrpValidationError(
            f"Customer and station movement_node_id overlap: {overlap[:10]}"
            f"{'…' if len(overlap) > 10 else ''}."
        )
    if depot_nid in customer_node_ids:
        raise EvrpValidationError(
            f"Depot node {depot_nid} is also used by a customer."
        )
    if depot_nid in station_node_ids:
        raise EvrpValidationError(
            f"Depot node {depot_nid} is also used by a station."
        )

    if len(set(service_nodes)) != len(service_nodes):
        dup = len(service_nodes) - len(set(service_nodes))
        raise EvrpValidationError(
            f"Duplicate movement_node_id entries in service_nodes ({dup} duplicate indices)."
        )

"""
Backward-compatible entry points for instance generation.

``generate_instance`` dispatches to the correct variant pipeline based on
``config.variant``.  The phased API (``generate_customers_phase`` ->
``generate_stations_phase`` -> ``finalize_benchmark_instance``) still works
for the classic EVRPTW variant and delegates to :mod:`variants.classic`.

For new variants, use the variant-specific entry points directly:
    - ``variants.classic.generate_classic_evrptw``
    - ``variants.multi_depot.generate_multi_depot_evrptw``
    - ``variants.two_echelon.generate_two_echelon_evrp``
"""

from typing import Any, Dict, List, Optional

from .exceptions import EvrpUserError
from .types import (
    BenchmarkInstance,
    CustomerPhaseState,
    EVFeatures,
    GenerationConfig,
    PipelineState,
    StationRecord,
)


# ── Phased API (classic EVRPTW only, backward compat) ────────────────────

def generate_customers_phase(
    config: GenerationConfig,
    ev_features: Optional[EVFeatures] = None,
    movement_graph: Any = None,
) -> CustomerPhaseState:
    """
    Build the movement graph, snap the depot, and produce the customer list.

    Delegates to :mod:`variants.classic`.
    Returns a ``CustomerPhaseState`` for backward compatibility.
    """
    from .variants.classic import prepare_graph_and_depot, generate_customers

    if ev_features is None:
        ev_features = EVFeatures()

    state = prepare_graph_and_depot(config, ev_features, movement_graph)
    state = generate_customers(state)

    return CustomerPhaseState(
        config=state.config,
        ev_features=state.ev_features,
        movement_graph=state.movement_graph,
        rng=state.rng,
        disk_cache=state.disk_cache,
        depot_node_id=state.depot_node_id,
        depot_snap_dist_m=state.depot_snap_dist_m,
        bbox=state.bbox,
        depot_to_node_time=state.depot_to_node_time,
        customers=state.customers,
    )


def generate_stations_phase(state: CustomerPhaseState) -> List[StationRecord]:
    """
    Fetch OSM candidates, snap, and select stations using the customer set.

    Delegates to :mod:`variants.classic`.
    """
    from .variants.classic import generate_stations

    ps = PipelineState(
        config=state.config,
        ev_features=state.ev_features,
        movement_graph=state.movement_graph,
        rng=state.rng,
        disk_cache=state.disk_cache,
        bbox=state.bbox,
        depot_node_id=state.depot_node_id,
        depot_snap_dist_m=state.depot_snap_dist_m,
        depot_to_node_time=state.depot_to_node_time,
        customers=state.customers,
    )
    ps = generate_stations(ps)
    return ps.stations


def finalize_benchmark_instance(
    state: CustomerPhaseState,
    stations: List[StationRecord],
    *,
    compute_matrices: bool = True,
    run_energy_feasibility: bool = True,
) -> BenchmarkInstance:
    """
    Order service nodes, compute matrices, run feasibility, return instance.

    Delegates to :mod:`variants.classic`.
    """
    from .variants.classic import finalize

    ps = PipelineState(
        config=state.config,
        ev_features=state.ev_features,
        movement_graph=state.movement_graph,
        rng=state.rng,
        disk_cache=state.disk_cache,
        bbox=state.bbox,
        depot_node_id=state.depot_node_id,
        depot_snap_dist_m=state.depot_snap_dist_m,
        depot_to_node_time=state.depot_to_node_time,
        customers=state.customers,
        stations=stations,
    )
    return finalize(ps, compute_matrices=compute_matrices, run_energy_feasibility=run_energy_feasibility)


# ── Unified dispatcher ────────────────────────────────────────────────────

def generate_instance(
    config: GenerationConfig,
    ev_features: Optional[EVFeatures] = None,
    movement_graph: Any = None,
    compute_matrices: bool = True,
    run_energy_feasibility: bool = True,
) -> BenchmarkInstance:
    """
    Main entry point — dispatches to the correct variant pipeline based on
    ``config.variant``.

    Default behaviour (``variant="classic_evrptw"``) is fully backward
    compatible with the original single-variant library.

    **Depots (all variants):** ``config.depot_lat`` / ``depot_lon`` identify the **facility**
    (building / warehouse site). Each pipeline snaps that location to the nearest
    legal drivable graph node for shortest-path computation; vehicles are modeled
    as entering and leaving the facility through that road access node.

    Parameters
    ----------
    config : GenerationConfig
        Must include ``variant`` to select the pipeline.
    ev_features : EVFeatures, optional
        Vehicle physics.  Defaults to standard EV.
    movement_graph : optional
        Pre-loaded graph to skip OSM download.
    compute_matrices : bool
        Attach pairwise matrices to the instance.
    run_energy_feasibility : bool
        Run energy-aware feasibility even when matrices aren't stored.
    """
    variant = config.variant

    if variant == "classic_evrptw":
        from .variants.classic import generate_classic_evrptw
        return generate_classic_evrptw(
            config, ev_features, movement_graph,
            compute_matrices, run_energy_feasibility,
        )

    if variant == "multi_depot_evrptw":
        from .variants.multi_depot import generate_multi_depot_evrptw
        return generate_multi_depot_evrptw(
            config, ev_features, movement_graph,
            compute_matrices, run_energy_feasibility,
        )

    if variant == "two_echelon_evrp":
        from .variants.two_echelon import generate_two_echelon_evrp
        return generate_two_echelon_evrp(
            config, ev_features, movement_graph,
            compute_matrices, run_energy_feasibility,
        )

    if variant == "pickup_delivery_evrp":
        raise EvrpUserError(
            "Variant 'pickup_delivery_evrp' has been removed. "
            "Use classic_evrptw, multi_depot_evrptw, or two_echelon_evrp."
        )
    raise EvrpUserError(
        f"Unknown variant {variant!r}. Supported: "
        f"classic_evrptw, multi_depot_evrptw, two_echelon_evrp."
    )


__all__ = [
    "finalize_benchmark_instance",
    "generate_customers_phase",
    "generate_instance",
    "generate_stations_phase",
]

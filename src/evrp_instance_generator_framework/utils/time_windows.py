"""
Tightness-aware time-window assignment.

Each tightness level controls window width bounds, safety margin, and repair
behaviour. The anchor method (reference travel time → temporal anchor) is
unchanged from the paper Appendix A.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np

from ..types import (
    CustomerCandidate,
    CustomerRecord,
    GenerationConfig,
    GenerationRepairSummary,
    TimeWindowTightness,
)


class TWProfile(NamedTuple):
    """Resolved time-window parameters for a given tightness level."""

    delta_minus_s: int
    delta_plus_s: int
    safety_buffer_s: int
    repair_margin_s: int
    minimum_width_s: int
    maximum_width_s: int
    service_to_window_warn_ratio: float = 0.6


# Paper-style absolute bounds (depot 08:00–17:00 compatible).
_ABSOLUTE_TW_PROFILES: Dict[TimeWindowTightness, Dict[str, int]] = {
    "wide": {"min_width_s": 7200, "max_width_s": 14400, "safety_margin_s": 900},
    "medium": {"min_width_s": 5400, "max_width_s": 10800, "safety_margin_s": 600},
    "tight": {"min_width_s": 3600, "max_width_s": 7200, "safety_margin_s": 300},
}

_TIGHTNESS_FRACTIONS: Dict[TimeWindowTightness, Tuple[float, float, float, float, float]] = {
    #                   delta_minus  delta_plus  safety  repair  min_width
    "wide":   (0.15, 0.15, 0.02, 0.04, 0.06),
    "medium": (0.06, 0.06, 0.01, 0.02, 0.02),
    "tight":  (0.02, 0.02, 0.005, 0.01, 0.01),
}


def resolve_tw_profile(config: GenerationConfig) -> TWProfile:
    """
    Build the concrete TW profile for *config.time_window_tightness*.

    Fraction-based deltas are combined with absolute min/max width bounds from
    the paper profiles (whichever is stricter for minimum width).
    """
    tightness = config.time_window_tightness
    depot_range_s = max(1, config.depot_time_close_s - config.depot_time_open_s)
    fracs = _TIGHTNESS_FRACTIONS[tightness]
    abs_prof = _ABSOLUTE_TW_PROFILES[tightness]
    min_w = max(int(fracs[4] * depot_range_s), int(abs_prof["min_width_s"]))
    max_w = min(int(depot_range_s), int(abs_prof["max_width_s"]))
    return TWProfile(
        delta_minus_s=int(fracs[0] * depot_range_s),
        delta_plus_s=int(fracs[1] * depot_range_s),
        safety_buffer_s=int(abs_prof["safety_margin_s"]),
        repair_margin_s=int(fracs[3] * depot_range_s),
        minimum_width_s=min_w,
        maximum_width_s=max_w,
    )


def _normalize(x: float, x_min: float, x_max: float) -> float:
    if x_max <= x_min:
        return 0.5
    return (x - x_min) / (x_max - x_min)


def _record_rejection(repair: Optional[GenerationRepairSummary], reason: str) -> None:
    if repair is None:
        return
    repair.customer_rejection_reasons[reason] = (
        repair.customer_rejection_reasons.get(reason, 0) + 1
    )


def validate_customer_time_fields(
    record: CustomerRecord,
    config: GenerationConfig,
    tw_profile: TWProfile,
    repair: Optional[GenerationRepairSummary] = None,
) -> None:
    """Assert post-assignment customer time fields (raises on hard violations)."""
    if record.time_open_s >= record.time_close_s:
        raise ValueError(f"Customer {record.id}: time_open_s >= time_close_s")
    if record.service_time_s <= 0:
        raise ValueError(f"Customer {record.id}: service_time_s must be > 0")
    if record.parking_time_s < 0:
        raise ValueError(f"Customer {record.id}: parking_time_s must be >= 0")
    if record.time_close_s > config.depot_time_close_s:
        raise ValueError(f"Customer {record.id}: time_close_s exceeds depot close")
    if record.time_open_s < config.depot_time_open_s:
        raise ValueError(f"Customer {record.id}: time_open_s before depot open")
    width = record.time_close_s - record.time_open_s
    ratio = (record.service_time_s + record.parking_time_s) / max(1, width)
    if ratio > tw_profile.service_to_window_warn_ratio and repair is not None:
        repair.tight_window_warnings += 1


def assign_time_window(
    candidate: CustomerCandidate,
    depot_travel_time_s: float,
    depot_tt_min_s: float,
    depot_tt_max_s: float,
    config: GenerationConfig,
    tw_profile: TWProfile,
    rng: np.random.Generator,
    repair_summary: Optional[GenerationRepairSummary] = None,
) -> CustomerRecord:
    """
    Assign demand, service time, and a tightness-aware time window.

    Repair logic (Appendix A style): widen/shift within depot horizon when the
    initial window cannot fit service + parking + safety margin.
    """
    demand = int(rng.integers(config.demand_min, config.demand_max + 1))
    service_time_s = int(config.service_time_base_s + config.service_time_per_unit_s * demand)
    parking_time_s = int(config.parking_time_s)

    norm = _normalize(depot_travel_time_s, depot_tt_min_s, depot_tt_max_s)
    temporal_anchor = config.depot_time_open_s + norm * (
        config.depot_time_close_s - config.depot_time_open_s
    )

    time_open_s = int(max(config.depot_time_open_s, temporal_anchor - tw_profile.delta_minus_s))
    time_close_s = int(min(config.depot_time_close_s, temporal_anchor + tw_profile.delta_plus_s))

    earliest_arrival_s = config.depot_time_open_s + float(depot_travel_time_s)
    needed_finish_s = earliest_arrival_s + service_time_s + parking_time_s + tw_profile.safety_buffer_s

    repaired = False
    if time_close_s < needed_finish_s:
        repaired = True
        time_close_s = int(
            min(
                config.depot_time_close_s,
                max(needed_finish_s, earliest_arrival_s + service_time_s + tw_profile.repair_margin_s),
            )
        )
        if time_close_s - time_open_s > tw_profile.maximum_width_s:
            time_open_s = int(max(config.depot_time_open_s, time_close_s - tw_profile.maximum_width_s))

    if time_open_s >= time_close_s:
        repaired = True
        time_close_s = int(
            min(config.depot_time_close_s, max(int(needed_finish_s), config.depot_time_open_s + tw_profile.minimum_width_s))
        )
        time_open_s = int(
            max(config.depot_time_open_s, time_close_s - tw_profile.minimum_width_s)
        )

    if time_open_s >= time_close_s:
        _record_rejection(repair_summary, "time_window_repair_impossible")
        raise ValueError(
            f"Cannot assign feasible time window for candidate {candidate.id} "
            f"(depot horizon too tight for travel + service)."
        )

    width = time_close_s - time_open_s
    if width < tw_profile.minimum_width_s:
        repaired = True
        time_open_s = int(max(config.depot_time_open_s, time_close_s - tw_profile.minimum_width_s))
        if time_open_s >= time_close_s:
            _record_rejection(repair_summary, "minimum_width_violation")
            raise ValueError(f"Minimum time-window width violated for candidate {candidate.id}.")

    if repaired and repair_summary is not None:
        repair_summary.time_window_repairs += 1

    record = CustomerRecord(
        id=candidate.id,
        lat=candidate.lat,
        lon=candidate.lon,
        movement_node_id=candidate.movement_node_id,
        snap_distance_m=candidate.snap_distance_m,
        demand=demand,
        service_time_s=service_time_s,
        parking_time_s=parking_time_s,
        time_open_s=time_open_s,
        time_close_s=time_close_s,
    )
    validate_customer_time_fields(record, config, tw_profile, repair_summary)
    return record

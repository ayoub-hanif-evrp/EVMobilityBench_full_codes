"""
Three-tier feasibility reporting: structural validity, time windows, energy (optional).

Used by variant ``finalize`` steps; results are stored on ``BenchmarkInstance.feasibility``.
"""

from __future__ import annotations

from .suite import (
    SCHEMA_VERSION,
    REPORT_MODE,
    build_classic_report,
    build_multi_depot_report,
    build_two_echelon_report,
)

__all__ = [
    "SCHEMA_VERSION",
    "REPORT_MODE",
    "build_classic_report",
    "build_multi_depot_report",
    "build_two_echelon_report",
]

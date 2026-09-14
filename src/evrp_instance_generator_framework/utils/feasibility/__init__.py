"""
Compatibility shim: feasibility checks live in ``evrp_benchmark.feasibility_tests``.
"""

from ...feasibility_tests import (
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

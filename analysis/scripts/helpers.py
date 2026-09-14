"""Shared helpers for reading instance-level result CSVs."""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from paths import DATA_DIR

CITY_ORDER = ("Casablanca", "Madrid", "Paris", "Shenzhen")
PATTERN_LABEL = {"c": "Clustered", "rc": "Mixed", "r": "Dispersed"}
PATTERN_ORDER = ("c", "rc", "r")
WONG = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#000000"]


def read_csv(name: str, folder: Path | None = None) -> list[dict]:
    path = (folder or DATA_DIR) / name
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fnum(row: dict, key: str) -> float:
    try:
        v = float(row[key])
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def classic_matched_pattern_rows(rows: Sequence[dict]) -> list[dict]:
    classic = [r for r in rows if r.get("variant") == "classic_evrptw"]
    buckets: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in classic:
        key = (r["city"], str(r["instance_seed"]), str(r["n_customers"]), str(r["n_stations"]))
        buckets[key][r["pattern"]] = r
    out: list[dict] = []
    for d in buckets.values():
        if set(d) >= {"c", "r", "rc"}:
            out.extend(d[p] for p in ("c", "r", "rc"))
    return out

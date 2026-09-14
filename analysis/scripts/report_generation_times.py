"""
Summarize benchmark-generation performance from generation_time_by_design.csv.

Reads the released CSV only; does not rerun the performance campaign.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from helpers import CITY_ORDER, fnum, read_csv
from paths import DATA_DIR

CSV_NAME = "generation_time_by_design.csv"
VARIANT_ORDER = ("classic_evrptw", "multi_depot_evrptw", "two_echelon_evrp")


def report_generation_times() -> list[dict]:
    rows = read_csv(CSV_NAME)
    if not rows:
        raise FileNotFoundError(f"Missing CSV: {DATA_DIR / CSV_NAME}")

    print(
        f"[generation-time] Mean generation times (s) from {CSV_NAME} "
        "(separate performance campaign, warm-cache)",
        flush=True,
    )
    print(
        f"{'city':12s} {'variant':20s} {'pattern':8s} "
        f"{'mean_s':>10s} {'std_s':>10s} {'n':>4s}",
        flush=True,
    )
    out: list[dict] = []
    for city in CITY_ORDER:
        for variant in VARIANT_ORDER:
            for pattern in ("c", "r", "rc"):
                match = [
                    r for r in rows
                    if r.get("city") == city
                    and r.get("instance_type") == variant
                    and r.get("customer_pattern") == pattern
                ]
                if not match:
                    continue
                r = match[0]
                mean_s = fnum(r, "generation_time_s_mean")
                std_s = fnum(r, "generation_time_s_std")
                n = int(float(r.get("generation_time_s_count") or 0))
                rec = {
                    "city": city,
                    "instance_type": variant,
                    "customer_pattern": pattern,
                    "generation_time_s_mean": mean_s,
                    "generation_time_s_std": std_s,
                    "generation_time_s_count": n,
                }
                out.append(rec)
                print(
                    f"{city:12s} {variant:20s} {pattern:8s} "
                    f"{mean_s:10.3f} {std_s:10.3f} {n:4d}",
                    flush=True,
                )
    print(f"[generation-time] reported {len(out)} design rows", flush=True)
    return out


if __name__ == "__main__":
    report_generation_times()

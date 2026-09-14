"""
Regenerate Figures 1-8 (+ Fig_S01) as PNG plots @ 600 dpi.

Visualisation only - reads existing CSVs; does not alter numerical values.

Design rule: every quantitative figure uses the same simple grammar so a reader
can decode it once and reuse it everywhere:
    rows = cities (or customer patterns), bar = median, thin line = IQR,
    the headline number is printed next to each bar.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common import city_country_map, load_graph
from helpers import (
    CITY_ORDER,
    PATTERN_LABEL,
    PATTERN_ORDER,
    WONG,
    classic_matched_pattern_rows,
    fnum,
    read_csv,
)
from paths import FIGURES_DIR

CITY_COLOR = {
    "Casablanca": WONG[0],
    "Madrid": WONG[1],
    "Paris": WONG[2],
    "Shenzhen": WONG[3],
}
PATTERN_COLOR = {"c": WONG[0], "rc": WONG[1], "r": WONG[2]}

BAR_ALPHA = 0.88
BAR_HEIGHT = 0.5
GRID_COLOR = "#DCDCDC"
TEXT_COLOR = "#1A1A1A"
NOTE_COLOR = "#767676"
IQR_NOTE = "bar: city median    line: interquartile range"

FIGURE_STEMS = (
    "Fig_01_framework_workflow",
    "Fig_02_road_networks",
    "Fig_03_distance_distortion",
    "Fig_04_directional_asymmetry",
    "Fig_05_charging_accessibility",
    "Fig_06_customer_pattern_validation",
    "Fig_07_travel_time_distortion",
    "Fig_08_energy_distortion",
    "Fig_S01_station_provenance",
)

# ---------------------------------------------------------------------------
# Published summary values used as regression checks. Plotted quantities are
# always computed from the CSVs; a mismatch aborts before writing the PNG.
# ---------------------------------------------------------------------------
EXPECTED_DISTANCE_RATIO = {"Casablanca": 1.175, "Madrid": 1.244, "Paris": 1.214, "Shenzhen": 1.251}
EXPECTED_DISTANCE_OVERALL = 1.226
EXPECTED_ASYM_PAIRS_PCT = {"Casablanca": 93.9, "Madrid": 97.8, "Paris": 98.6, "Shenzhen": 95.1}
EXPECTED_ASYM_PAIRS_OVERALL = 96.4
EXPECTED_ASYM_REL_PCT = {"Casablanca": 1.65, "Madrid": 3.36, "Paris": 5.02, "Shenzhen": 1.57}
EXPECTED_ASYM_REL_OVERALL = 2.69
EXPECTED_MISMATCH_PCT = {"Casablanca": 11.3, "Madrid": 18.1, "Paris": 20.6, "Shenzhen": 13.7}
EXPECTED_MISMATCH_OVERALL = 15.0
EXPECTED_DETOUR_MEDIAN_KM = {"Casablanca": 0.465, "Madrid": 1.403, "Paris": 0.451, "Shenzhen": 0.834}
EXPECTED_DETOUR_P90_KM = {"Casablanca": 7.799, "Madrid": 4.288, "Paris": 1.092, "Shenzhen": 1.784}
EXPECTED_NND_MEDIAN_M = {"Clustered": 116.7, "Mixed": 440.7, "Dispersed": 1756.3}
EXPECTED_NND_Q1_M = {"Clustered": 108.1, "Mixed": 275.6, "Dispersed": 1329.4}
EXPECTED_NND_Q3_M = {"Clustered": 132.3, "Mixed": 666.3, "Dispersed": 2675.2}
EXPECTED_NND_TRIPLES = 40
EXPECTED_TIME_RATIO = {"Casablanca": 1.187, "Madrid": 1.257, "Paris": 1.228, "Shenzhen": 1.265}
EXPECTED_TIME_OVERALL = 1.239
EXPECTED_ENERGY_RATIO = {"Casablanca": 1.165, "Madrid": 1.293, "Paris": 1.223, "Shenzhen": 1.266}
EXPECTED_ENERGY_OVERALL = 1.250
EXPECTED_NEG_ENERGY_OVERALL_PCT = 0.50
EXPECTED_PROVENANCE_CASABLANCA = {"observed": 43.1, "proxy": 56.8, "synthetic": 0.1}
EXPECTED_PROVENANCE_FULL_OBSERVED = ("Madrid", "Paris", "Shenzhen")

TOL_RATIO = 1e-3
TOL_PCT_1DP = 0.055
TOL_PCT_2DP = 0.01
TOL_KM = 1e-3
TOL_METRE = 0.055
TOL_PROV_PCT = 0.15


def _check(name: str, got: dict, expected: dict, tol: float) -> None:
    """Abort before writing a figure if a plotted value disagrees with the published summary."""
    bad = []
    for key, exp in expected.items():
        val = got.get(key)
        if val is None or not np.isfinite(val) or abs(float(val) - exp) > tol:
            bad.append(f"{key}: computed={val!r}  expected={exp}")
    if bad:
        raise ValueError(
            f"[{name}] does not match published summary values (tolerance {tol}); "
            "figure not written. Inspect the CSV column / aggregation:\n    "
            + "\n    ".join(bad)
        )


def _style(plt) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "axes.labelpad": 3.0,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 2.0,
    })


def _save(fig, stem: str, *, pad_inches: float = 0.02) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"{stem}.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=pad_inches)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  wrote {out.name}", flush=True)


def _ordered_groups(rows, group_key, metric, order):
    labels, data = [], []
    for g in order:
        col = [fnum(r, metric) for r in rows if r.get(group_key) == g]
        col = [x for x in col if np.isfinite(x)]
        if col:
            labels.append(g)
            data.append(col)
    return labels, data


def _summary(cols):
    meds, q1, q3 = [], [], []
    for col in cols:
        arr = np.asarray(col, dtype=float)
        meds.append(float(np.median(arr)))
        q1.append(float(np.percentile(arr, 25)))
        q3.append(float(np.percentile(arr, 75)))
    return np.asarray(meds), np.asarray(q1), np.asarray(q3)


def _panel(ax, text: str) -> None:
    ax.text(0.0, 1.02, text, transform=ax.transAxes, fontweight="bold",
            fontsize=8.5, va="bottom", ha="left")


def _note(fig, text: str) -> None:
    fig.text(0.995, 0.005, text, ha="right", va="bottom",
             fontsize=5.8, color=NOTE_COLOR)


def _rows_axis(ax, labels):
    """Top-down categorical rows with a light vertical grid."""
    y = np.arange(len(labels), dtype=float)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(len(labels) - 0.45, -0.55)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID_COLOR, lw=0.45, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return y


def _overall_line(ax, value: float, text: str) -> None:
    """Pooled median reference line across city rows."""
    ax.axvline(value, color="#3A3A3A", ls=(0, (3, 2)), lw=0.7, zorder=4)
    ax.text(value, 1.01, text, transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=6.0, color="#3A3A3A")


def _bar_rows(ax, labels, meds, q1=None, q3=None, *, base=0.0, label_fmt=None,
              colors=None, headroom=0.16):
    """Median bars per row + IQR line + printed value. The shared plot idiom."""
    y = _rows_axis(ax, labels)
    meds = np.asarray(meds, dtype=float)
    cols = colors if colors is not None else [CITY_COLOR[l] for l in labels]
    ax.barh(y, meds - base, left=base, height=BAR_HEIGHT, color=cols,
            alpha=BAR_ALPHA, linewidth=0, zorder=2)

    if q1 is not None:
        q1 = np.asarray(q1, dtype=float)
        q3 = np.asarray(q3, dtype=float)
        ax.hlines(y, q1, q3, color="#3A3A3A", lw=0.8, zorder=3)
        anchors = np.maximum(meds, q3)
    else:
        anchors = meds

    span = float(anchors.max()) - base
    ax.set_xlim(base, base + span * (1.0 + headroom))
    if label_fmt is not None:
        for i in range(len(labels)):
            ax.text(anchors[i] + 0.025 * span, y[i], label_fmt(meds[i]),
                    va="center", ha="left", fontsize=6.8, color=TEXT_COLOR, zorder=4)
    ax.axvline(base, color="#4A4A4A", lw=0.8, zorder=1)
    return y


def _matched_nnd_triples(pat_rows):
    """Return list of (c, rc, r) NND triples for matched classic instances."""
    buckets: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in pat_rows:
        key = (r["city"], str(r["instance_seed"]), str(r["n_customers"]), str(r["n_stations"]))
        buckets[key][r["pattern"]] = fnum(r, "median_nearest_neighbor_distance")
    triples = []
    for d in buckets.values():
        if set(d) >= {"c", "rc", "r"} and all(np.isfinite(d[p]) for p in ("c", "rc", "r")):
            triples.append((d["c"], d["rc"], d["r"]))
    return triples


# ---------------------------------------------------------------------------
# Fig 01 - workflow
# ---------------------------------------------------------------------------

def fig_01_framework_workflow() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    stages = [
        "Configuration\n& EV features",
        "Road-network\npreparation",
        "Facility &\ncustomer\ngeneration",
        "Charging-station\nplacement",
        "Service-graph\nconstruction",
        "Instance\nvalidation &\ndiagnostics",
        "Benchmark\ninstance",
    ]
    n = len(stages)
    fig, ax = plt.subplots(figsize=(7.4, 1.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    gap = 0.020
    box_w = (1.0 - (n + 1) * gap) / n
    box_h, y0 = 0.78, 0.11
    edge = "#2F4F5F"
    centres = []
    for i, label in enumerate(stages):
        x0 = gap + i * (box_w + gap)
        centres.append(x0 + box_w / 2)
        face = "#E4EFE6" if i == n - 1 else "#F1F5F8"
        ax.add_patch(FancyBboxPatch(
            (x0, y0), box_w, box_h,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            linewidth=0.8, edgecolor=edge, facecolor=face,
        ))
        ax.text(x0 + box_w / 2, y0 + box_h / 2, label, ha="center", va="center",
                fontsize=6.2, color="#111111", linespacing=1.3)
    for i in range(n - 1):
        ax.add_patch(FancyArrowPatch(
            (centres[i] + box_w / 2 + 0.003, y0 + box_h / 2),
            (centres[i + 1] - box_w / 2 - 0.003, y0 + box_h / 2),
            arrowstyle="-|>", mutation_scale=6, lw=0.7, color=edge,
        ))
    _save(fig, "Fig_01_framework_workflow", pad_inches=0.015)


# ---------------------------------------------------------------------------
# Fig 02 - road networks
# ---------------------------------------------------------------------------

def fig_02_road_networks() -> bool:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    cc = city_country_map()
    panel_aspect = 1.45  # width / height of every panel, so areas stay equal
    fig, axes = plt.subplots(2, 2, figsize=(5.9, 5.9 / panel_aspect))
    ok = False
    for ax, city, let in zip(axes.ravel(), CITY_ORDER, "abcd"):
        try:
            G = load_graph(city, cc[city])
        except Exception as exc:
            ax.text(0.5, 0.5, f"{city}\n(unavailable)", ha="center", va="center", fontsize=7)
            ax.set_axis_off()
            print(f"  map load failed {city}: {exc}", flush=True)
            continue
        ok = True
        segs = [[(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])]
                for u, v in G.edges()]
        ax.add_collection(LineCollection(segs, colors="#3A3A3A", linewidths=0.08))
        xs = [G.nodes[n]["x"] for n in G.nodes]
        ys = [G.nodes[n]["y"] for n in G.nodes]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        # Longitude degrees shrink with latitude: keep the map geographically
        # true, then pad the shorter side so all panels share one frame shape.
        kx = float(np.cos(np.deg2rad(cy)))
        dx, dy = (x1 - x0) * 1.02, (y1 - y0) * 1.02
        if (dx * kx) / dy > panel_aspect:
            dy = dx * kx / panel_aspect
        else:
            dx = panel_aspect * dy / kx
        ax.set_xlim(cx - dx / 2.0, cx + dx / 2.0)
        ax.set_ylim(cy - dy / 2.0, cy + dy / 2.0)
        ax.set_aspect(1.0 / kx)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.01, 0.99, f"({let}) {city}", transform=ax.transAxes,
                fontsize=7.5, fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.02, hspace=0.02)
    if ok:
        _save(fig, "Fig_02_road_networks", pad_inches=0.01)
    else:
        plt.close(fig)
    return ok


# ---------------------------------------------------------------------------
# Fig 03 / 07 / 08a - ratio distortion (shared layout)
# ---------------------------------------------------------------------------

def _pct_above_one(m: float) -> str:
    return f"+{(m - 1.0) * 100:.1f}%"


def _ratio_figure(rows, metric, *, stem, name, xlabel, expected, overall_expected) -> None:
    """Shared body of Fig. 3 and Fig. 7: city medians of a road/geometric ratio."""
    import matplotlib.pyplot as plt

    labels, data = _ordered_groups(rows, "city", metric, CITY_ORDER)
    meds, q1, q3 = _summary(data)
    pooled = float(np.median([fnum(r, metric) for r in rows
                              if np.isfinite(fnum(r, metric))]))
    _check(name, dict(zip(labels, meds)), expected, TOL_RATIO)
    _check(name, {"overall": pooled}, {"overall": overall_expected}, TOL_RATIO)

    fig, ax = plt.subplots(figsize=(4.4, 2.0))
    _bar_rows(ax, labels, meds, q1, q3, base=1.0, label_fmt=_pct_above_one,
              headroom=0.18)
    _overall_line(ax, pooled, f"overall {pooled:.3f}")
    ax.set_xlabel(xlabel)
    fig.subplots_adjust(left=0.21, right=0.98, top=0.90, bottom=0.26)
    _note(fig, IQR_NOTE)
    _save(fig, stem)


def fig_03_distance_distortion(dist) -> None:
    _ratio_figure(
        dist, "median_ratio",
        stem="Fig_03_distance_distortion",
        name="Fig. 3 distance ratio",
        xlabel=r"Road / geometric distance   $d_{\mathrm{road}}/d_{\mathrm{geo}}$",
        expected=EXPECTED_DISTANCE_RATIO, overall_expected=EXPECTED_DISTANCE_OVERALL,
    )


def fig_07_travel_time_distortion(tt) -> None:
    _ratio_figure(
        tt, "median_time_ratio",
        stem="Fig_07_travel_time_distortion",
        name="Fig. 7 travel-time ratio",
        xlabel=r"Road / geometric travel-time baseline   $t_{\mathrm{road}}/t_{\mathrm{geo}}$",
        expected=EXPECTED_TIME_RATIO, overall_expected=EXPECTED_TIME_OVERALL,
    )


# ---------------------------------------------------------------------------
# Fig 04 - directional asymmetry
# ---------------------------------------------------------------------------

def fig_04_directional_asymmetry(dire) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.1))
    specs = [
        ("fraction_asymmetric_pairs", "Asymmetric directed pairs (%)", "(a)",
         EXPECTED_ASYM_PAIRS_PCT, EXPECTED_ASYM_PAIRS_OVERALL, TOL_PCT_1DP, 1),
        ("median_asymmetry", "Median relative asymmetry (%)", "(b)",
         EXPECTED_ASYM_REL_PCT, EXPECTED_ASYM_REL_OVERALL, TOL_PCT_2DP, 2),
    ]
    for ax, (metric, xlabel, lab, expected, overall_exp, tol, digits) in zip(axes, specs):
        labels, data = _ordered_groups(dire, "city", metric, CITY_ORDER)
        meds, q1, q3 = _summary([[100.0 * x for x in col] for col in data])
        pooled = 100.0 * float(np.median([fnum(r, metric) for r in dire
                                          if np.isfinite(fnum(r, metric))]))
        name = f"Fig. 4{lab[1]} {metric}"
        _check(name, dict(zip(labels, meds)), expected, tol)
        _check(name, {"overall": pooled}, {"overall": overall_exp}, tol)

        _bar_rows(ax, labels, meds, q1, q3, headroom=0.20,
                  label_fmt=lambda m, d=digits: f"{m:.{d}f}%")
        _overall_line(ax, pooled, f"overall {pooled:.{digits}f}%")
        ax.set_xlabel(xlabel)
        _panel(ax, lab)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.84, bottom=0.25, wspace=0.42)
    _note(fig, IQR_NOTE)
    _save(fig, "Fig_04_directional_asymmetry")


# ---------------------------------------------------------------------------
# Fig 05 - charging accessibility
# ---------------------------------------------------------------------------

def fig_05_charging_accessibility(ch) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.15))

    # (a) mismatch rate
    labels, data = _ordered_groups(ch, "city", "mismatch_rate", CITY_ORDER)
    meds, q1, q3 = _summary([[100.0 * x for x in col] for col in data])
    pooled = 100.0 * float(np.median([fnum(r, "mismatch_rate") for r in ch
                                      if np.isfinite(fnum(r, "mismatch_rate"))]))
    _check("Fig. 5a mismatch rate", dict(zip(labels, meds)), EXPECTED_MISMATCH_PCT, TOL_PCT_1DP)
    _check("Fig. 5a mismatch rate", {"overall": pooled},
           {"overall": EXPECTED_MISMATCH_OVERALL}, TOL_PCT_1DP)
    ax = axes[0]
    _bar_rows(ax, labels, meds, q1, q3, label_fmt=lambda m: f"{m:.1f}%", headroom=0.20)
    _overall_line(ax, pooled, f"overall {pooled:.1f}%")
    ax.set_xlabel("Nearest-station mismatch rate (%)")
    _panel(ax, "(a)")

    # (b) median and 90th percentile across instances of the instance-level
    # median conditional detour.
    ax = axes[1]
    cities = list(CITY_ORDER)
    med_det, p90_det = [], []
    for city in cities:
        vals = np.asarray([fnum(r, "median_conditional_detour_km") for r in ch
                           if r.get("city") == city], dtype=float)
        vals = vals[np.isfinite(vals)]
        med_det.append(float(np.median(vals)))
        p90_det.append(float(np.percentile(vals, 90)))
    med_det = np.asarray(med_det)
    p90_det = np.asarray(p90_det)
    _check("Fig. 5b median conditional detour", dict(zip(cities, med_det)),
           EXPECTED_DETOUR_MEDIAN_KM, TOL_KM)
    _check("Fig. 5b P90 conditional detour", dict(zip(cities, p90_det)),
           EXPECTED_DETOUR_P90_KM, TOL_KM)

    y = _rows_axis(ax, cities)
    hi = float(p90_det.max())
    ax.hlines(y, med_det, p90_det, color="#9A9A9A", lw=1.2, zorder=2)
    for i, city in enumerate(cities):
        ax.plot(med_det[i], y[i], "o", ms=4.4, color=CITY_COLOR[city],
                markeredgecolor="#222222", markeredgewidth=0.3, zorder=3)
        ax.plot(p90_det[i], y[i], "s", ms=4.0, color=CITY_COLOR[city],
                markeredgecolor="#222222", markeredgewidth=0.3, zorder=3)
        ax.text(med_det[i] - 0.030 * hi, y[i], f"{med_det[i]:.3f}", va="center", ha="right",
                fontsize=6.3, color=TEXT_COLOR)
        ax.text(p90_det[i] + 0.030 * hi, y[i], f"{p90_det[i]:.3f}", va="center", ha="left",
                fontsize=6.3, color=TEXT_COLOR)
    ax.set_xlim(-0.20 * hi, 1.22 * hi)
    ax.set_xlabel("Instance-level median conditional detour (km)")
    # Neutral grey keys: shape encodes statistic; city colour is only on the data.
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555",
                   markeredgecolor="#222222", markersize=5, label="City median"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#555555",
                   markeredgecolor="#222222", markersize=4.6,
                   label="90th percentile across instances"),
    ]
    ax.legend(
        handles=handles, frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, -0.32), ncol=1, fontsize=5.8,
        handletextpad=0.35, borderpad=0.0, labelspacing=0.2,
        columnspacing=0.8, handlelength=1.0,
    )
    _panel(ax, "(b)")

    fig.subplots_adjust(left=0.13, right=0.98, top=0.84, bottom=0.36, wspace=0.40)
    _note(fig,
          "(a) bar: city median; line: interquartile range    "
          "(b) circle: city median; square: 90th percentile across instances")
    _save(fig, "Fig_05_charging_accessibility")


# ---------------------------------------------------------------------------
# Fig 06 - customer pattern validation
# ---------------------------------------------------------------------------

def fig_06_customer_pattern_validation(pat) -> None:
    import matplotlib.pyplot as plt

    triples = np.asarray(_matched_nnd_triples(pat), dtype=float)
    labels = [PATTERN_LABEL[p] for p in PATTERN_ORDER]
    meds = np.median(triples, axis=0)
    q1 = np.percentile(triples, 25, axis=0)
    q3 = np.percentile(triples, 75, axis=0)
    colors = [PATTERN_COLOR[p] for p in PATTERN_ORDER]

    _check("Fig. 6 matched triples", {"n": triples.shape[0]},
           {"n": EXPECTED_NND_TRIPLES}, 0)
    _check("Fig. 6 NND median", dict(zip(labels, meds)), EXPECTED_NND_MEDIAN_M, TOL_METRE)
    _check("Fig. 6 NND Q1", dict(zip(labels, q1)), EXPECTED_NND_Q1_M, TOL_METRE)
    _check("Fig. 6 NND Q3", dict(zip(labels, q3)), EXPECTED_NND_Q3_M, TOL_METRE)

    fig, ax = plt.subplots(figsize=(4.4, 1.85))
    _bar_rows(ax, labels, meds, q1, q3, colors=colors, headroom=0.18,
              label_fmt=lambda m: f"{m:,.1f} m")
    ax.set_xlabel("Nearest-neighbour distance between customers (m)")
    fig.subplots_adjust(left=0.21, right=0.98, top=0.97, bottom=0.27)
    _note(fig, f"{triples.shape[0]} matched classical EVRPTW triples    "
               "bar: median    line: interquartile range")
    _save(fig, "Fig_06_customer_pattern_validation")


# ---------------------------------------------------------------------------
# Fig 08 - energy distortion
# ---------------------------------------------------------------------------

def fig_08_energy_distortion(en) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.1),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

    labels, data = _ordered_groups(en, "city", "median_energy_ratio", CITY_ORDER)
    meds, q1, q3 = _summary(data)
    pooled = float(np.median([fnum(r, "median_energy_ratio") for r in en
                              if np.isfinite(fnum(r, "median_energy_ratio"))]))
    _check("Fig. 8a energy ratio", dict(zip(labels, meds)), EXPECTED_ENERGY_RATIO, TOL_RATIO)
    _check("Fig. 8a energy ratio", {"overall": pooled},
           {"overall": EXPECTED_ENERGY_OVERALL}, TOL_RATIO)
    ax = axes[0]
    _bar_rows(ax, labels, meds, q1, q3, base=1.0, label_fmt=_pct_above_one, headroom=0.18)
    _overall_line(ax, pooled, f"overall {pooled:.3f}")
    ax.set_xlabel(r"Road / geometric energy baseline   $E_{\mathrm{road}}/E_{\mathrm{geo}}$")
    _panel(ax, "(a)")

    labels_n, data_n = _ordered_groups(en, "city", "fraction_negative_energy_arcs", CITY_ORDER)
    meds_n, q1_n, q3_n = _summary([[100.0 * x for x in col] for col in data_n])
    pooled_n = 100.0 * float(np.median([fnum(r, "fraction_negative_energy_arcs") for r in en
                                        if np.isfinite(fnum(r, "fraction_negative_energy_arcs"))]))
    _check("Fig. 8b negative energy arcs", {"overall": pooled_n},
           {"overall": EXPECTED_NEG_ENERGY_OVERALL_PCT}, TOL_PCT_2DP)
    ax = axes[1]
    _bar_rows(ax, labels_n, meds_n, q1_n, q3_n, label_fmt=lambda m: f"{m:.2f}%", headroom=0.20)
    _overall_line(ax, pooled_n, f"overall {pooled_n:.2f}%")
    ax.set_xlabel("Arcs with negative energy (%)")
    _panel(ax, "(b)")

    fig.subplots_adjust(left=0.14, right=0.98, top=0.84, bottom=0.25, wspace=0.42)
    _note(fig, IQR_NOTE)
    _save(fig, "Fig_08_energy_distortion")


# ---------------------------------------------------------------------------
# Fig S01 - station provenance
# ---------------------------------------------------------------------------

def fig_s01_station_provenance(prov) -> None:
    import matplotlib.pyplot as plt

    cities = list(CITY_ORDER)
    shares = {"observed": [], "proxy": [], "synthetic": []}
    for city in cities:
        rows = [r for r in prov if r["city"] == city]
        for key, col in (("observed", "fraction_observed"),
                         ("proxy", "fraction_proxy"),
                         ("synthetic", "fraction_synthetic")):
            shares[key].append(100.0 * float(np.mean([fnum(r, col) for r in rows])) if rows else 0.0)

    totals = {c: sum(shares[k][i] for k in shares) for i, c in enumerate(cities)}
    _check("Fig. S1 provenance shares sum to 100%", totals,
           {c: 100.0 for c in cities}, 0.05)
    casa = {
        "observed": shares["observed"][cities.index("Casablanca")],
        "proxy": shares["proxy"][cities.index("Casablanca")],
        "synthetic": shares["synthetic"][cities.index("Casablanca")],
    }
    _check("Fig. S1 Casablanca provenance", casa, EXPECTED_PROVENANCE_CASABLANCA, TOL_PROV_PCT)
    for city in EXPECTED_PROVENANCE_FULL_OBSERVED:
        i = cities.index(city)
        _check(
            f"Fig. S1 {city} observed EV",
            {"observed": shares["observed"][i], "proxy": shares["proxy"][i],
             "synthetic": shares["synthetic"][i]},
            {"observed": 100.0, "proxy": 0.0, "synthetic": 0.0},
            TOL_PROV_PCT,
        )

    fig, ax = plt.subplots(figsize=(4.6, 1.85))
    y = _rows_axis(ax, cities)
    ax.grid(False)
    segments = [
        ("observed", "Observed EV", WONG[0]),
        ("proxy", "Proxy host", WONG[1]),
        ("synthetic", "Synthetic", WONG[2]),
    ]
    left = np.zeros(len(cities))
    for key, label, color in segments:
        vals = np.asarray(shares[key])
        ax.barh(y, vals, left=left, height=BAR_HEIGHT, color=color, alpha=BAR_ALPHA,
                linewidth=0, label=label, zorder=2)
        for i, v in enumerate(vals):
            if v >= 8.0:
                ax.text(left[i] + v / 2.0, y[i], f"{v:.1f}%", ha="center", va="center",
                        fontsize=6.5, color="white", fontweight="bold", zorder=3)
        left = left + vals
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of selected charging stations (%)")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              fontsize=6.8, handlelength=1.1, handletextpad=0.4, columnspacing=1.2)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.84, bottom=0.27)
    syn = shares["synthetic"]
    _note(fig, f"Synthetic share: {syn[0]:.1f}% in Casablanca, "
               f"{max(syn[1:]):.1f}% in Madrid, Paris and Shenzhen")
    _save(fig, "Fig_S01_station_provenance")


def generate_figures() -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    dist = read_csv("distance_instance.csv")
    dire = read_csv("direction_instance.csv")
    ch = read_csv("charging_instance.csv")
    pat = classic_matched_pattern_rows(read_csv("customer_pattern_metrics.csv"))
    en = read_csv("energy_instance.csv")
    tt = read_csv("travel_time_instance.csv")
    prov = read_csv("station_provenance.csv")

    written = []
    print("[figures] regenerating figures …", flush=True)

    fig_01_framework_workflow(); written.append("Fig_01_framework_workflow")
    if fig_02_road_networks(): written.append("Fig_02_road_networks")
    if dist:
        fig_03_distance_distortion(dist); written.append("Fig_03_distance_distortion")
    if dire:
        fig_04_directional_asymmetry(dire); written.append("Fig_04_directional_asymmetry")
    if ch:
        fig_05_charging_accessibility(ch); written.append("Fig_05_charging_accessibility")
    if pat:
        fig_06_customer_pattern_validation(pat); written.append("Fig_06_customer_pattern_validation")
    if tt:
        fig_07_travel_time_distortion(tt); written.append("Fig_07_travel_time_distortion")
    if en:
        fig_08_energy_distortion(en); written.append("Fig_08_energy_distortion")
    if prov:
        fig_s01_station_provenance(prov); written.append("Fig_S01_station_provenance")

    for pdf in FIGURES_DIR.glob("*.pdf"):
        pdf.unlink()

    missing = [s for s in FIGURE_STEMS if not (FIGURES_DIR / f"{s}.png").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing figures: {missing}")
    print("[figures] all figure stems present; regression checks passed", flush=True)
    print(f"[figures] wrote {len(written)} figures -> {FIGURES_DIR}", flush=True)
    return {"stems": written}


if __name__ == "__main__":
    generate_figures()

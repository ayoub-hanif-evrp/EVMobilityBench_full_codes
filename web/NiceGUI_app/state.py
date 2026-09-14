"""Mutable wizard state for NiceGUI app (single-user local session)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


@dataclass(frozen=True)
class MapEmbed:
    src: str = ""
    height_px: int = 520

    def __bool__(self) -> bool:
        return bool(self.src)

from evrp_instance_generator_framework.types import GenerationConfig


_DEFAULT_DEPOT_SNAP_M = float(GenerationConfig.__dataclass_fields__["depot_snap_max_dist_m"].default)


@dataclass
class WizardState:
    step: int = 1
    busy_message: str = ""
    # Non-empty: full-body transition spinner (moving between wizard steps).
    step_transition_caption: str = ""
    # Non-empty while Step 3 map is hidden and a facility snap is in progress.
    facility_snap_busy: str = ""
    status_message: str = ""
    status_level: str = "info"

    city: str = "Casablanca"
    country: str = "Morocco"

    variant: str = "classic_evrptw"
    seed: int = 1234
    num_stations: int = 5
    energy_period: str = "off_peak"
    use_elevation: bool = True
    num_customers: int = 20
    num_clusters: int = 3
    pattern: str = "rc"
    demand_min: int = 5
    demand_max: int = 20
    tw_tightness: str = "medium"
    num_additional_depots: int = 2
    two_echelon_num_satellites: int = 3
    cluster_max_radius_m: float = 2500.0
    cluster_min_separation_m: float = 100.0
    customer_building_osm_min_candidates: int = 0
    station_osm_min_candidates: int = 0
    feasibility_scope: str = "time_only"

    customer_csv_bytes: bytes | None = None

    battery_kwh: float = 75.0
    mass_kg: float = 1800.0
    rolling_f: float = 0.01
    mass_factor: float = 1.05
    drag_cd: float = 0.29
    frontal_m2: float = 2.2
    driver_behavior: str = "passive"
    heating: bool = False
    cooling: bool = False
    rain: bool = False

    movement_graph: Any = None
    depot_node_id: int | None = None
    depot_facility_lat: float | None = None
    depot_facility_lon: float | None = None
    instance: Any = None

    md_depot_lat: list[float] = field(default_factory=list)
    md_depot_lon: list[float] = field(default_factory=list)
    md_depot_node: list[int | None] = field(default_factory=list)
    extra_depots: list[tuple[float, float]] = field(default_factory=list)

    export_fmt: str = "json"
    export_selection: dict[str, bool] = field(default_factory=dict)

    map_embed_depot: MapEmbed = field(default_factory=MapEmbed)
    map_embed_result: MapEmbed = field(default_factory=MapEmbed)
    map_embed_overview: MapEmbed = field(default_factory=MapEmbed)

    def n_facilities(self) -> int:
        if self.variant == "multi_depot_evrptw":
            return 1 + max(0, int(self.num_additional_depots))
        if self.variant == "two_echelon_evrp":
            return 1 + max(1, int(self.two_echelon_num_satellites))
        return 1

    def ensure_facility_arrays(self, center_lat: float, center_lon: float) -> None:
        n = self.n_facilities()
        while len(self.md_depot_lat) < n:
            self.md_depot_lat.append(center_lat)
        while len(self.md_depot_lon) < n:
            self.md_depot_lon.append(center_lon)
        while len(self.md_depot_node) < n:
            self.md_depot_node.append(None)
        self.md_depot_lat = self.md_depot_lat[:n]
        self.md_depot_lon = self.md_depot_lon[:n]
        self.md_depot_node = self.md_depot_node[:n]

    def cust_params_dict(self) -> dict[str, Any]:
        cust: dict[str, Any] = {
            "seed": int(self.seed),
            "num_stations": int(self.num_stations),
            "energy_period": self.energy_period,
            "use_elevation": self.use_elevation,
            "variant": self.variant,
            "num_customers": int(self.num_customers),
            "num_clusters": int(self.num_clusters),
            "customer_pattern": self.pattern,
            "demand_min": int(self.demand_min),
            "demand_max": int(self.demand_max),
            "time_window_tightness": self.tw_tightness,
            "customer_csv_bytes": self.customer_csv_bytes,
        }
        if self.pattern in ("c", "rc"):
            cust["cluster_max_radius_m"] = None if self.cluster_max_radius_m <= 0 else float(self.cluster_max_radius_m)
            cust["cluster_min_separation_m"] = (
                None if self.cluster_min_separation_m <= 0 else float(self.cluster_min_separation_m)
            )
        if self.variant == "multi_depot_evrptw":
            cust["num_additional_depots"] = int(self.num_additional_depots)
        if self.variant == "two_echelon_evrp":
            cust["two_echelon_num_satellites"] = int(self.two_echelon_num_satellites)
        if self.customer_building_osm_min_candidates > 0:
            cust["customer_building_osm_min_candidates"] = int(self.customer_building_osm_min_candidates)
        if self.station_osm_min_candidates > 0:
            cust["station_osm_min_candidates"] = int(self.station_osm_min_candidates)
        return cust

    def ev_params_dict(self) -> dict[str, Any]:
        return {
            "battery_capacity_kwh": float(self.battery_kwh),
            "mass_kg": float(self.mass_kg),
            "rolling_resistance_coeff_f": float(self.rolling_f),
            "mass_factor_delta": float(self.mass_factor),
            "drag_coefficient_cd": float(self.drag_cd),
            "frontal_area_m2": float(self.frontal_m2),
            "driver_behavior": self.driver_behavior,
            "heating_on": 1 if self.heating else 0,
            "cooling_on": 1 if self.cooling else 0,
            "raining_on": 1 if self.rain else 0,
        }

    @staticmethod
    def default_depot_snap_m() -> float:
        return _DEFAULT_DEPOT_SNAP_M

    def reset_to_initial(self) -> None:
        fresh = WizardState()
        for f in fields(WizardState):
            setattr(self, f.name, getattr(fresh, f.name))

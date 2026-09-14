from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

# ---------------------------------------------------------------------------
# Shared type aliases
# ---------------------------------------------------------------------------
Period = Literal["off_peak", "midday", "pm_peak"]
CustomerPattern = Literal["c", "r", "rc"]
StationType = Literal["fast", "normal"]
DriverBehavior = Literal["passive", "aggressive"]

# New type aliases for the multi-variant framework
EVRPVariant = Literal[
    "classic_evrptw",
    "multi_depot_evrptw",
    "two_echelon_evrp",
]
TimeWindowTightness = Literal["wide", "medium", "tight"]
FeasibilityLevel = Literal[
    "local_screening",
    "constructive_feasibility",
    "exact_small_instance_check",
    "validity_time_energy",
]
StationSourceType = Literal["observed_ev", "proxy_host", "synthetic"]


# ---------------------------------------------------------------------------
# EV Features (unchanged)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EVFeatures:
    """
    Vehicle physics parameters used by the generator's energy equations.

    The defaults represent a "standard EV" for benchmarking.
    """

    battery_capacity_kwh: float = 75.0

    mass_kg: float = 1800.0
    rolling_resistance_coeff_f: float = 0.01
    mass_factor_delta: float = 1.05
    drag_coefficient_cd: float = 0.29
    frontal_area_m2: float = 2.2
    air_density_kg_m3: float = 1.225

    driver_behavior: DriverBehavior = "passive"

    heating_on: int = 0
    cooling_on: int = 0
    raining_on: int = 0

    @property
    def speed_multiplier(self) -> float:
        return 0.95 if self.driver_behavior == "passive" else 1.05


# ---------------------------------------------------------------------------
# Generation config — extended for all variants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GenerationConfig:
    """
    All user-controllable inputs for the benchmark generator.

    Default values produce backward-compatible *classic EVRPTW* behaviour.
    Set ``variant`` to enable multi-depot or two-echelon.

    **Depot model (every variant):** ``depot_lat`` / ``depot_lon`` are the **facility**
    (e.g. warehouse / building footprint or address geocode). The pipeline snaps
    that point to the nearest drivable ``movement_graph`` node; routing and matrices
    use that node as where the vehicle joins the road network to serve the facility.
    """

    # -- Variant selector (default = classic) --
    variant: EVRPVariant = "classic_evrptw"

    # -- Location --
    city: str = ""
    country: str = ""

    # Depot **facility** (WGS-84): building footprint centroid, geocoded address, or map pin.
    # Always snapped to the nearest drivable ``movement_graph`` node for routing matrices.
    depot_lat: float = 0.0
    depot_lon: float = 0.0
    depot_snap_max_dist_m: float = 500.0

    # -- Randomness --
    seed: int = 1234

    # -- Customer generation --
    # If customer_csv_path is set, you can leave num_customers as None.
    # The framework will derive the customer count from the CSV.
    num_customers: Optional[int] = 50
    customer_csv_path: Optional[str] = None
    customer_pattern: CustomerPattern = "rc"
    num_clusters: int = 5
    customers_pool_snap_max_dist_m: float = 150.0
    cluster_max_radius_m: Optional[float] = 2500.0
    cluster_min_separation_m: Optional[float] = 100.0

    demand_min: int = 5
    demand_max: int = 20
    service_time_base_s: int = 120
    service_time_per_unit_s: int = 20
    parking_time_s: int = 60

    depot_time_open_s: int = 8 * 3600
    depot_time_close_s: int = 17 * 3600

    # -- Time-window tightness (new — first-class benchmark parameter) --
    time_window_tightness: TimeWindowTightness = "medium"

    # Legacy TW parameters kept for backward compat; overridden by tightness
    # when ``time_window_tightness`` is set and the variant pipeline uses
    # the new tightness-aware assignment.
    delta_minus_s: int = 30 * 60
    delta_plus_s: int = 30 * 60
    safety_buffer_s: int = 5 * 60
    repair_margin_s: int = 10 * 60
    minimum_time_window_width_s: int = 10 * 60

    anchor_period: Period = "off_peak"
    travel_time_periods: Tuple[Period, Period, Period] = (
        "off_peak",
        "midday",
        "pm_peak",
    )

    # -- Stations --
    num_stations: int = 8
    real_stations_snap_max_dist_m: float = 150.0
    depot_weight_for_station_coverage: float = 0.5

    station_fast_fraction: float = 0.3
    green_station_fraction: float = 0.3
    fast_charging_power_kW: float = 120.0
    normal_charging_power_kW: float = 50.0
    default_min_slots: int = 2

    # -- Energy / feasibility --
    energy_period: Period = "off_peak"
    feasibility_max_customers_one_hop_via_station: bool = True
    # Stored on exported metadata; generation always runs ``feasibility_tests`` (validity / time / energy).
    feasibility_level: FeasibilityLevel = "validity_time_energy"

    # -- Graph --
    node_elevation_provider: Literal["srtm", "open_elevation", "none"] = "srtm"
    # OSMnx ``graph_from_place`` network. Default ``drive`` = public roads only (drops
    # most ``highway=service``, reducing **port / pier** spurs). Use ``drive_service``
    # when you need alleys / dense old-city links (more OSM edges).
    osm_network_type: Literal["drive", "drive_service"] = "drive"
    # When True, OSMnx keeps every weakly connected fragment inside the place polygon
    # before the largest-SCC step in ``prepare_movement_graph``.
    osm_retain_all: bool = True

    # -- OSM pool sizes --
    customer_building_osm_min_candidates: Optional[int] = None
    station_osm_min_candidates: Optional[int] = None

    # -- Disk cache --
    osm_cache_enabled: bool = True
    osm_cache_dir: Optional[str] = None

    # =====================================================================
    # Multi-depot EVRPTW fields (ignored when variant != multi_depot_evrptw)
    # =====================================================================
    # Manual mode: provide explicit (lat, lon) tuples.
    # Auto mode:   leave additional_depots empty and set num_additional_depots > 0
    #              to let the system place depots algorithmically.
    additional_depots: Tuple[Tuple[float, float], ...] = ()
    num_additional_depots: int = 2
    additional_depot_snap_max_dist_m: Optional[float] = None
    additional_depot_time_open_s: Optional[int] = None
    additional_depot_time_close_s: Optional[int] = None

    # =====================================================================
    # Two-echelon EVRP fields (ignored when variant != two_echelon_evrp)
    # =====================================================================
    # Manual mode: provide explicit (lat, lon) tuples.
    # Auto mode:   leave satellite_locations empty and set num_satellites > 0
    #              to let the system place satellites algorithmically.
    satellite_locations: Tuple[Tuple[float, float], ...] = ()
    num_satellites: int = 3
    satellite_snap_max_dist_m: float = 200.0
    satellite_capacity: Optional[int] = None


# ---------------------------------------------------------------------------
# Record types — customers
# ---------------------------------------------------------------------------
@dataclass
class CustomerRecord:
    id: int
    lat: float
    lon: float
    movement_node_id: int
    snap_distance_m: float

    demand: int
    service_time_s: int
    parking_time_s: int
    time_open_s: int
    time_close_s: int


@dataclass(frozen=True)
class CustomerCandidate:
    """A snapped OSM building candidate (no demand/time window assigned yet)."""

    id: int
    lat: float
    lon: float
    movement_node_id: int
    snap_distance_m: float


# ---------------------------------------------------------------------------
# Record types — stations (extended with provenance)
# ---------------------------------------------------------------------------
@dataclass
class StationRecord:
    id: int
    lat: float
    lon: float
    movement_node_id: int
    snap_distance_m: float

    time_open_s: int
    time_close_s: int
    number_slots: int
    station_type: StationType
    charging_power_kW: float
    charging_price_per_kWh: float
    green_label: int
    source: str  # backward compat: "real" or "synthetic"

    # Provenance (new — always populated by the framework)
    station_source_type: StationSourceType = "synthetic"
    source_priority: int = 3
    is_real_observed_ev: bool = False
    osm_tags: Optional[Dict[str, Any]] = field(default=None, repr=False)


@dataclass(frozen=True)
class StationCandidate:
    """A snapped OSM station candidate (no slot/power/price assigned yet)."""

    id: int
    lat: float
    lon: float
    movement_node_id: int
    snap_distance_m: float
    is_green_hint: Optional[int] = None
    charging_power_kW_hint: Optional[float] = None
    num_slots_hint: Optional[int] = None

    # Provenance tracking (set during extraction)
    station_source_type: StationSourceType = "observed_ev"
    source_priority: int = 1
    osm_tags: Optional[Dict[str, Any]] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Record types — multi-depot
# ---------------------------------------------------------------------------
@dataclass
class DepotRecord:
    """A depot in a multi-depot instance.

    ``facility_lat`` / ``facility_lon`` are the **declared depot site** (building centroid,
    user pin, or OSM feature centroid) in WGS-84.

    ``lat`` / ``lon`` are the **snapped** coordinates of ``movement_node_id`` on the drive
    graph — used consistently with shortest-path matrices.

    ``id`` 0 corresponds to ``GenerationConfig.depot_lat`` / ``depot_lon`` (synthesis order only);
    all depots are equivalent for routing. ``is_primary`` is kept for backward compatibility and
    is not used to privilege a depot in the pipeline.
    """

    id: int
    lat: float
    lon: float
    facility_lat: float
    facility_lon: float
    movement_node_id: int
    snap_distance_m: float
    time_open_s: int
    time_close_s: int
    is_primary: bool = False


def depot_facility_latlon(dep: DepotRecord) -> Tuple[float, float]:
    """
    (latitude, longitude) of the depot **facility** for maps and exports.

    Falls back to snapped ``lat`` / ``lon`` when ``facility_*`` is missing
    (e.g. instances held in Streamlit session from an older package version).
    """
    la = getattr(dep, "facility_lat", None)
    lo = getattr(dep, "facility_lon", None)
    if la is not None and lo is not None:
        return float(la), float(lo)
    return float(dep.lat), float(dep.lon)


# ---------------------------------------------------------------------------
# Record types — two-echelon
# ---------------------------------------------------------------------------
@dataclass
class SatelliteRecord:
    """
    An intermediate transfer facility between first and second echelons.

    Satellites receive goods from depot(s) via first-echelon vehicles and
    distribute them to end-customers via second-echelon vehicles.

    ``lat`` / ``lon`` are the **facility** site (user pin or OSM feature centroid);
    ``movement_node_id`` is the snapped **drivable** access node for routing on
    ``movement_graph`` (same enter/leave-via-nearest-road pattern as the depot).
    """

    id: int
    lat: float
    lon: float
    movement_node_id: int
    snap_distance_m: float
    capacity: int
    time_open_s: int
    time_close_s: int
    assigned_customer_ids: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline state (phased API)
# ---------------------------------------------------------------------------
@dataclass
class CustomerPhaseState:
    """
    Result after OSM building selection and customer record creation.

    Use :func:`evrp_benchmark.generator.generate_stations_phase` next.
    """

    config: GenerationConfig
    ev_features: EVFeatures
    movement_graph: Any
    rng: Any
    disk_cache: Any
    depot_node_id: int
    depot_snap_dist_m: float
    bbox: Tuple[float, float, float, float]
    depot_to_node_time: Dict[int, float]
    customers: List[CustomerRecord]


@dataclass
class GenerationRepairSummary:
    """Counters for generation repairs and rejections (audit trail only)."""

    customer_resamples: int = 0
    station_resamples: int = 0
    time_window_repairs: int = 0
    duplicate_rejections: int = 0
    tight_window_warnings: int = 0
    customer_rejection_reasons: Dict[str, int] = field(default_factory=dict)


@dataclass
class PipelineState:
    """
    General-purpose intermediate state shared across variant pipelines.

    Each variant populates only the fields it needs; the others stay at
    their defaults.
    """

    config: GenerationConfig
    ev_features: EVFeatures
    movement_graph: Any = None
    rng: Any = None
    disk_cache: Any = None
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    # Depot(s)
    depot_node_id: int = -1
    depot_snap_dist_m: float = 0.0
    depot_to_node_time: Dict[int, float] = field(default_factory=dict)
    depots: List[DepotRecord] = field(default_factory=list)
    # Multi-depot only: index-aligned with ``depots`` (forward / return travel labels).
    depot_travel_times: List[Dict[int, float]] = field(default_factory=list, repr=False)
    depot_return_times: List[Dict[int, float]] = field(default_factory=list, repr=False)

    customers: List[CustomerRecord] = field(default_factory=list)

    # Satellites (two-echelon)
    satellites: List[SatelliteRecord] = field(default_factory=list)

    # Stations
    stations: List[StationRecord] = field(default_factory=list)

    # Unified extraction results (populated once, consumed by customer + station phases)
    _unified_buildings: List[Any] = field(default_factory=list, repr=False)
    _unified_ev_stations: List[Any] = field(default_factory=list, repr=False)
    _unified_proxy_hosts: List[Any] = field(default_factory=list, repr=False)
    _unified_synthetic_hosts: List[Any] = field(default_factory=list, repr=False)
    _unified_extracted: bool = False

    repair_summary: GenerationRepairSummary = field(default_factory=GenerationRepairSummary)


# ---------------------------------------------------------------------------
# Instance metadata — extended for variant awareness
# ---------------------------------------------------------------------------
@dataclass
class InstanceMetadata:
    city: str
    country: str
    seed: int
    movement_node_count: int
    service_node_count: int

    # Variant-aware metadata (new)
    variant: EVRPVariant = "classic_evrptw"
    time_window_tightness: TimeWindowTightness = "medium"
    feasibility_level: FeasibilityLevel = "validity_time_energy"
    depot_count: int = 1
    satellite_count: int = 0
    customer_count: int = 0
    station_count_observed_ev: int = 0
    station_count_proxy_host: int = 0
    station_count_synthetic: int = 0
    elevation_enabled: bool = True
    two_echelon_enabled: bool = False

    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Benchmark instance — the final, researcher-consumable object
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkInstance:
    """
    Final, researcher-consumable object.

    The depot **facility** (building / site) is ``config.depot_lat`` / ``config.depot_lon``.
    ``depot_node_id`` is the **road access** vertex: shortest-path matrices and travel
    times assume the vehicle reaches the facility via that nearest legal drivable node.
    Multi-depot instances also store per-depot ``facility_*`` on each ``DepotRecord``.
    """

    metadata: InstanceMetadata
    config: GenerationConfig

    movement_graph: Any  # networkx.DiGraph
    service_nodes: List[int]  # depot, customers..., stations...

    customers: List[CustomerRecord]
    stations: List[StationRecord]
    depot_node_id: int

    # Optional matrices (None when compute_matrices=False)
    distance_matrix_m: Any = None
    travel_time_matrices_s: Dict[Period, Any] = field(default_factory=dict)
    energy_matrix_kwh: Any = None

    feasibility: Dict[str, Any] = field(default_factory=dict)

    generation_report: Dict[str, Any] = field(default_factory=dict)

    # Multi-depot (empty list for classic single-depot)
    depots: List[DepotRecord] = field(default_factory=list)

    # Two-echelon (empty list for non-2E variants)
    satellites: List[SatelliteRecord] = field(default_factory=list)


def primary_depot_facility_latlon(instance: BenchmarkInstance) -> Tuple[float, float]:
    """
    (latitude, longitude) of the **primary depot facility** (building / declared site)
    for maps and reporting.

    When ``instance.depots`` is non-empty (multi-depot), uses depot ``id`` 0.
    Otherwise uses ``config.depot_lat`` / ``config.depot_lon`` (classic, two-echelon).
    """
    if instance.depots:
        d0 = instance.depots[0]
        for d in instance.depots:
            if d.id == 0:
                d0 = d
                break
        return depot_facility_latlon(d0)
    c = instance.config
    return float(c.depot_lat), float(c.depot_lon)

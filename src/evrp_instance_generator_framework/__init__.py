"""
EVRP benchmark instance generation on OSM-derived urban road networks.

**Variants supported:**
    - Classic EVRPTW (default)
    - Multi-depot EVRPTW
    - Two-echelon EVRP

**Quick start:** ``generate_instance(config)`` — dispatches by ``config.variant``.

**Depots (all variants):** ``GenerationConfig.depot_lat``/``depot_lon`` are the **facility**
(building / warehouse site); each pipeline snaps them to the nearest **drivable** graph
node so shortest paths model access via that road vertex.

**Phased (classic):** ``generate_customers_phase`` -> ``generate_stations_phase``
-> ``finalize_benchmark_instance``.

**Variant-specific:** ``generate_classic_evrptw``, ``generate_multi_depot_evrptw``,
``generate_two_echelon_evrp``.
"""

# -- Core types --
from .types import (
    BenchmarkInstance,
    CustomerCandidate,
    CustomerPhaseState,
    CustomerRecord,
    DepotRecord,
    EVFeatures,
    GenerationConfig,
    InstanceMetadata,
    PipelineState,
    SatelliteRecord,
    StationCandidate,
    StationRecord,
    depot_facility_latlon,
    primary_depot_facility_latlon,
)

# -- Exceptions --
from .exceptions import EvrpUserError, format_exception_for_user

# -- Generator API --
from .generator import (
    finalize_benchmark_instance,
    generate_customers_phase,
    generate_instance,
    generate_stations_phase,
)

# -- Variant-specific entry points --
from .variants.classic import generate_classic_evrptw
from .variants.multi_depot import generate_multi_depot_evrptw, suggest_additional_depot_facilities
from .variants.two_echelon import generate_two_echelon_evrp, suggest_satellite_facility_latlons

# -- Composable building blocks --
from .road_network.utils import (
    compute_service_matrices,
    download_road_network,
    build_disk_cache as make_disk_cache,
    movement_graph_bbox,
    prepare_movement_graph,
)
from .utils.snapping import snap_latlon_to_road
from .customers import (
    REQUIRED_CUSTOMER_COLUMNS,
    apply_customers_to_state,
    load_customers_from_csv,
    resolve_num_customers_from_config,
)

# -- Visualization --
from .visualization import (
    display_feasibility_summary,
    geographic_center_of_graph,
    graph_center_latlon,
    load_prepared_graph,
    plot_benchmark_on_map,
    plot_city_roads,
    plot_city_roads_with_depot,
    plot_services_on_map,
    prepare_city_road_network,
    print_feasibility_report,
    save_instance_overview_map,
    save_road_network_figure,
    save_road_network_with_markers,
)

# Interactive Folium maps are lazy so `evrp.map_*_interactive(...)` works even when the
# installed top-level namespace is stale; first access loads `.visualization`.
_INTERACTIVE_MAP_NAMES = frozenset(
    {
        "map_benchmark_interactive",
        "map_city_roads_interactive",
        "map_city_roads_with_depot_interactive",
        "map_services_interactive",
    }
)


def __getattr__(name: str):
    if name in _INTERACTIVE_MAP_NAMES:
        from . import visualization as _viz

        return getattr(_viz, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# -- Export --
from .export.graph_to_geodataframes import (
    movement_graph_to_edges_gdf,
    movement_graph_to_nodes_gdf,
)

# -- Service graph utilities --
from .service_graph.service_node_mapping import build_service_nodes

# -- Generation primitives (advanced users) --
from .feasibility_tests import (
    SCHEMA_VERSION,
    REPORT_MODE,
    build_classic_report,
    build_multi_depot_report,
    build_two_echelon_report,
)
from .utils.time_windows import TWProfile, resolve_tw_profile

__all__ = [
    # Types
    "BenchmarkInstance",
    "CustomerCandidate",
    "CustomerPhaseState",
    "CustomerRecord",
    "DepotRecord",
    "EVFeatures",
    "GenerationConfig",
    "InstanceMetadata",
    "PipelineState",
    "SatelliteRecord",
    "StationCandidate",
    "StationRecord",
    "depot_facility_latlon",
    "primary_depot_facility_latlon",
    "TWProfile",
    # Exceptions
    "EvrpUserError",
    "format_exception_for_user",
    # Generator (dispatcher)
    "generate_instance",
    "generate_customers_phase",
    "generate_stations_phase",
    "finalize_benchmark_instance",
    # Variant entry points
    "generate_classic_evrptw",
    "generate_multi_depot_evrptw",
    "suggest_additional_depot_facilities",
    "generate_two_echelon_evrp",
    "suggest_satellite_facility_latlons",
    # Building blocks
    "compute_service_matrices",
    "download_road_network",
    "make_disk_cache",
    "movement_graph_bbox",
    "prepare_movement_graph",
    "snap_latlon_to_road",
    "REQUIRED_CUSTOMER_COLUMNS",
    "apply_customers_to_state",
    "load_customers_from_csv",
    "resolve_num_customers_from_config",
    # Visualization
    "display_feasibility_summary",
    "geographic_center_of_graph",
    "graph_center_latlon",
    "load_prepared_graph",
    "map_benchmark_interactive",
    "map_city_roads_interactive",
    "map_city_roads_with_depot_interactive",
    "map_services_interactive",
    "plot_benchmark_on_map",
    "plot_city_roads",
    "plot_city_roads_with_depot",
    "plot_services_on_map",
    "prepare_city_road_network",
    "print_feasibility_report",
    "save_instance_overview_map",
    "save_road_network_figure",
    "save_road_network_with_markers",
    # Export
    "movement_graph_to_edges_gdf",
    "movement_graph_to_nodes_gdf",
    # Service graph
    "build_service_nodes",
    # Generation primitives
    "resolve_tw_profile",
    "SCHEMA_VERSION",
    "REPORT_MODE",
    "build_classic_report",
    "build_multi_depot_report",
    "build_two_echelon_report",
]

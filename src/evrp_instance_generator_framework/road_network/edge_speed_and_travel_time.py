from typing import Any, Dict, Optional, Tuple

import networkx as nx


SPEED_LIMIT_FALLBACK: Dict[str, float] = {
    "motorway": 100,
    "trunk": 80,
    "primary": 60,
    "secondary": 50,
    "tertiary": 40,
    "residential": 30,
    "service": 20,
}

FREE_FLOW_FACTOR: Dict[str, float] = {
    "motorway": 0.90,
    "trunk": 0.90,
    "primary": 0.85,
    "secondary": 0.80,
    "tertiary": 0.80,
    "residential": 0.75,
    "service": 0.65,
}

TRAFFIC_MULTIPLIER: Dict[str, Dict[str, float]] = {
    "motorway": {"off_peak": 1.05, "midday": 1.15, "pm_peak": 1.30},
    "trunk": {"off_peak": 1.05, "midday": 1.15, "pm_peak": 1.30},
    "primary": {"off_peak": 1.08, "midday": 1.20, "pm_peak": 1.40},
    "secondary": {"off_peak": 1.08, "midday": 1.18, "pm_peak": 1.35},
    "tertiary": {"off_peak": 1.05, "midday": 1.12, "pm_peak": 1.25},
    "residential": {"off_peak": 1.02, "midday": 1.08, "pm_peak": 1.15},
    "service": {"off_peak": 1.00, "midday": 1.05, "pm_peak": 1.10},
}

MIN_SPEED_KPH = 5.0


def normalize_highway(highway_value: Any) -> Optional[str]:
    """
    OSM sometimes provides `highway` as a list; keep one class.
    """

    if highway_value is None:
        return None
    if isinstance(highway_value, list) and highway_value:
        return str(highway_value[0])
    return str(highway_value)


def parse_maxspeed(maxspeed_value: Any) -> Optional[float]:
    """
    Minimal parsing: if maxspeed is numeric -> float, or a string starting with digits.
    """

    if maxspeed_value is None:
        return None
    if isinstance(maxspeed_value, (int, float)):
        return float(maxspeed_value)
    if isinstance(maxspeed_value, str):
        stripped = maxspeed_value.strip()
        # common formats: "50 mph", "50", "signals"
        num = ""
        for ch in stripped:
            if ch.isdigit():
                num += ch
            else:
                break
        if num:
            return float(num)
    return None


def get_speed_limit_kph(edge_data: Dict[str, Any]) -> float:
    road_type = normalize_highway(edge_data.get("highway"))
    maxspeed = parse_maxspeed(edge_data.get("maxspeed"))

    if maxspeed is not None:
        return float(maxspeed)
    if road_type in SPEED_LIMIT_FALLBACK:
        return float(SPEED_LIMIT_FALLBACK[road_type])
    return 30.0


def get_free_flow_speed_kph(speed_limit_kph: float, road_type: Optional[str]) -> float:
    factor = FREE_FLOW_FACTOR.get(road_type, 0.75)
    return speed_limit_kph * factor


def get_effective_speed_kph(free_flow_speed_kph: float, road_type: Optional[str], period: str) -> Tuple[float, float]:
    m = TRAFFIC_MULTIPLIER.get(road_type, {}).get(period, 1.10)
    v = free_flow_speed_kph / m
    return max(v, MIN_SPEED_KPH), m


def travel_time_seconds(length_m: float, speed_kph: float) -> float:
    speed_mps = speed_kph * 1000.0 / 3600.0
    if speed_mps <= 0:
        raise ValueError("Speed must be positive.")
    return float(length_m / speed_mps)


def assign_edge_attributes(G: nx.DiGraph) -> nx.DiGraph:
    """
    Apply your TXT edge-speed & travel-time rules to every edge.

    Expects node attribute `elevation_m` to be attached beforehand.
    """

    for u, v, data in G.edges(data=True):
        road_type = normalize_highway(data.get("highway"))
        length_m = float(data.get("length", 0.0))
        if length_m <= 0:
            # Avoid missing weight attributes later in Dijkstra; use a small epsilon.
            length_m = 1.0

        speed_limit_kph = get_speed_limit_kph(data)
        free_flow_speed_kph = get_free_flow_speed_kph(speed_limit_kph, road_type)

        off_peak_speed_kph, off_peak_mult = get_effective_speed_kph(free_flow_speed_kph, road_type, "off_peak")
        midday_speed_kph, midday_mult = get_effective_speed_kph(free_flow_speed_kph, road_type, "midday")
        pm_peak_speed_kph, pm_peak_mult = get_effective_speed_kph(free_flow_speed_kph, road_type, "pm_peak")

        data["speed_limit_kph"] = speed_limit_kph
        data["free_flow_speed_kph"] = free_flow_speed_kph
        data["length_m"] = length_m

        data["off_peak_multiplier"] = off_peak_mult
        data["midday_multiplier"] = midday_mult
        data["pm_peak_multiplier"] = pm_peak_mult

        data["off_peak_speed_kph"] = off_peak_speed_kph
        data["midday_speed_kph"] = midday_speed_kph
        data["pm_peak_speed_kph"] = pm_peak_speed_kph

        data["off_peak_travel_time_s"] = travel_time_seconds(length_m, off_peak_speed_kph)
        data["midday_travel_time_s"] = travel_time_seconds(length_m, midday_speed_kph)
        data["pm_peak_travel_time_s"] = travel_time_seconds(length_m, pm_peak_speed_kph)

        # Elevation: compute slope angle (rad) from node elevations.
        # Positive = uphill (u→v gains altitude), negative = downhill.
        elev_u = float(G.nodes[u].get("elevation_m", 0.0))
        elev_v = float(G.nodes[v].get("elevation_m", 0.0))
        dz = elev_v - elev_u
        import math
        data["slope_angle_rad"] = math.atan2(dz, length_m)

    return G


"""
On-disk cache for OSM / Overpass downloads (GeoDataFrames and road graphs).

Same progressive pipelines and stopping rules as before; only successful
``ox.features_from_*`` / ``ox.graph_from_place`` results are persisted so
repeat runs for the same city/bbox skip network I/O.

Environment:
  EVRP_BENCHMARK_CACHE_DIR — override cache root directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import osmnx as _ox
_ox.settings.max_query_area_size = 50 * 1_000_000_000

FEATURE_CACHE_VERSION = 1
GRAPH_CACHE_VERSION = 2

_MISSING = object()


def default_cache_dir() -> Path:
    env = os.environ.get("EVRP_BENCHMARK_CACHE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "evrp_benchmark" / "cache"
    return Path.home() / ".cache" / "evrp_benchmark"


def _rounded_bbox(bbox: Optional[Tuple[float, float, float, float]]) -> Optional[Tuple[float, ...]]:
    if bbox is None:
        return None
    return tuple(round(float(x), 6) for x in bbox)


def _tags_fingerprint(tags: dict) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in sorted(tags.keys(), key=lambda x: str(x)):
        v = tags[k]
        if isinstance(v, bool):
            out[str(k)] = v
        elif v is None:
            out[str(k)] = None
        else:
            out[str(k)] = str(v)
    return out


def _feature_cache_key(
    namespace: str,
    tags: dict,
    bbox: Optional[Tuple[float, float, float, float]],
    city: str,
    country: str,
) -> str:
    payload = {
        "v": FEATURE_CACHE_VERSION,
        "ns": namespace,
        "tags": _tags_fingerprint(tags),
        "bbox": _rounded_bbox(bbox),
        "place": None
        if bbox is not None
        else f"{city.strip().casefold()}|{country.strip().casefold()}",
    }
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:28]
    return f"feat_{namespace}_{h}.pkl"


def _graph_cache_key(city: str, country: str, *, graph_profile: str = "default") -> str:
    payload = {
        "v": GRAPH_CACHE_VERSION,
        "city": city.strip().casefold(),
        "country": country.strip().casefold(),
        "graph_profile": graph_profile,
    }
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:28]
    return f"drive_graph_{h}.pkl"


class OsmDiskCache:
    """Pickle-based cache under ``features/`` and ``graphs/`` subfolders."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.features_dir = self.root / "features"
        self.graphs_dir = self.root / "graphs"
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.graphs_dir.mkdir(parents=True, exist_ok=True)

    def _feature_path(self, namespace: str, tags: dict, bbox, city: str, country: str) -> Path:
        return self.features_dir / _feature_cache_key(namespace, tags, bbox, city, country)

    def get_feature_gdf(self, namespace: str, tags: dict, bbox, city: str, country: str) -> Any:
        path = self._feature_path(namespace, tags, bbox, city, country)
        if not path.is_file():
            return _MISSING
        with open(path, "rb") as f:
            return pickle.load(f)

    def put_feature_gdf(
        self,
        namespace: str,
        tags: dict,
        bbox,
        city: str,
        country: str,
        gdf: Any,
    ) -> None:
        path = self._feature_path(namespace, tags, bbox, city, country)
        with open(path, "wb") as f:
            pickle.dump(gdf, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_graph(self, city: str, country: str, *, graph_profile: str = "default") -> Union[object, Any]:
        path = self.graphs_dir / _graph_cache_key(city, country, graph_profile=graph_profile)
        if not path.is_file():
            return _MISSING
        with open(path, "rb") as f:
            return pickle.load(f)

    def put_graph(self, city: str, country: str, graph: Any, *, graph_profile: str = "default") -> None:
        path = self.graphs_dir / _graph_cache_key(city, country, graph_profile=graph_profile)
        with open(path, "wb") as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)


def overpass_features_cached(
    tags: dict,
    bbox: Optional[Tuple[float, float, float, float]],
    city: str,
    country: str,
    disk_cache: Optional[OsmDiskCache],
    namespace: str,
):
    import osmnx as ox

    if disk_cache is not None:
        hit = disk_cache.get_feature_gdf(namespace, tags, bbox, city, country)
        if hit is not _MISSING:
            return hit

    try:
        if bbox is not None:
            gdf = ox.features_from_bbox(bbox=bbox, tags=tags)
        else:
            gdf = ox.features_from_place(f"{city}, {country}", tags=tags)
    except Exception:
        return None

    if disk_cache is not None:
        disk_cache.put_feature_gdf(namespace, tags, bbox, city, country, gdf)
    return gdf


def try_load_drive_graph(
    city: str,
    country: str,
    disk_cache: Optional[OsmDiskCache],
    *,
    graph_profile: str = "default",
) -> Optional[Any]:
    if disk_cache is None:
        return None
    g = disk_cache.get_graph(city, country, graph_profile=graph_profile)
    if g is _MISSING:
        return None
    return g


def save_drive_graph(
    city: str,
    country: str,
    graph: Any,
    disk_cache: Optional[OsmDiskCache],
    *,
    graph_profile: str = "default",
) -> None:
    if disk_cache is None:
        return
    disk_cache.put_graph(city, country, graph, graph_profile=graph_profile)

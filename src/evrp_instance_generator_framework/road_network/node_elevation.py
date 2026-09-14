from typing import Dict, List, Literal, Optional, Tuple

import logging
import os
import time
import zipfile
from pathlib import Path

import networkx as nx
import numpy as np

_log = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF_S = 2.0
_SRTM3_SAMPLES = 1201  # 3 arc-second → 1201×1201 grid per tile
# Same layout as USGS SRTM3; USGS occasionally 404s — mirror is a fallback.
_SRTM3_USGS_BASE = "https://dds.cr.usgs.gov/srtm/version2_1/SRTM3"
_SRTM3_MIRROR_BASE = "https://terrain.ardupilot.org/SRTM3"

ElevationProvider = Literal["srtm", "open_elevation", "none"]


def _srtm_tile_name(lat_floor: int, lon_floor: int) -> str:
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    return f"{ns}{abs(lat_floor):02d}{ew}{abs(lon_floor):03d}"


def _srtm_cache_dir() -> Path:
    env = os.environ.get("SRTM3_DIR")
    if env:
        return Path(env).expanduser().resolve()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "evrp_benchmark" / "srtm3"
    return Path.home() / ".cache" / "evrp_benchmark" / "srtm3"


def _continent_for_tile(lat: int, lon: int) -> str:
    """
    Continent subdirectory name for SRTM3 zip layout (USGS and common mirrors).

    The northern Mediterranean / Maghreb band (≥35°N, 15°W–55°E) uses
    ``Eurasia``, not ``Africa`` — matching that avoids spurious 404s (e.g. N36E002).
    """
    if (-12 <= lat < 60 and 55 <= lon < 180) or (35 <= lat < 61 and -15 <= lon < 55):
        return "Eurasia"
    if -35 <= lat < 40 and -20 <= lon < 55:
        return "Africa"
    if 15 <= lat < 62 and -170 <= lon < -50:
        return "North_America"
    if -60 <= lat < 15 and -90 <= lon < -30:
        return "South_America"
    if -50 <= lat < -5 and 100 <= lon < 180:
        return "Australia"
    return "Islands"


_tile_cache: Dict[str, Optional[np.ndarray]] = {}


def _load_srtm_tile(lat_floor: int, lon_floor: int) -> Optional[np.ndarray]:
    """Load an SRTM3 .hgt tile into a numpy array.  Returns None if unavailable."""
    name = _srtm_tile_name(lat_floor, lon_floor)
    if name in _tile_cache:
        return _tile_cache[name]

    cache_dir = _srtm_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    hgt_path = cache_dir / f"{name}.hgt"
    zip_path = cache_dir / f"{name}.hgt.zip"

    # Extract from zip if needed
    if not hgt_path.exists() and zip_path.exists():
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extract(f"{name}.hgt", cache_dir)
        except Exception as exc:
            _log.debug("Failed to extract %s: %s", zip_path, exc)

    # Auto-download if not present
    if not hgt_path.exists():
        import urllib.request

        continent = _continent_for_tile(lat_floor, lon_floor)
        sources = [
            ("USGS", f"{_SRTM3_USGS_BASE}/{continent}/{name}.hgt.zip"),
            ("SRTM mirror", f"{_SRTM3_MIRROR_BASE}/{continent}/{name}.hgt.zip"),
        ]
        _log.info("Downloading SRTM tile %s …", name)
        last_exc: Optional[BaseException] = None
        for label, url in sources:
            try:
                urllib.request.urlretrieve(url, zip_path)
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extract(f"{name}.hgt", cache_dir)
                if label != "USGS":
                    _log.info("SRTM tile %s loaded from %s.", name, label)
                break
            except Exception as exc:
                last_exc = exc
                if zip_path.exists():
                    try:
                        zip_path.unlink()
                    except OSError:
                        pass
                _log.debug("SRTM %s download failed for %s: %s", label, name, exc)
        else:
            _log.warning(
                "SRTM download failed for %s: %s — those nodes will use Open-Elevation API.",
                name, last_exc,
            )
            _tile_cache[name] = None
            return None

    if not hgt_path.exists():
        _tile_cache[name] = None
        return None

    expected = _SRTM3_SAMPLES * _SRTM3_SAMPLES * 2
    raw = hgt_path.read_bytes()
    if len(raw) != expected:
        _log.warning("SRTM tile %s has unexpected size %d (expected %d)", name, len(raw), expected)
        _tile_cache[name] = None
        return None

    tile = np.frombuffer(raw, dtype=">i2").reshape((_SRTM3_SAMPLES, _SRTM3_SAMPLES)).astype(np.float64)
    tile[tile == -32768] = 0.0  # SRTM void fill
    _tile_cache[name] = tile
    return tile


def _batch_srtm_elevations(lats: np.ndarray, lons: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorised SRTM3 tile lookup.

    Returns
    -------
    elevations : ndarray, shape (N,)
        Heights from tiles where available; 0 elsewhere.
    missing : ndarray, shape (N,), dtype bool
        True where the 1°×1° tile could not be loaded (caller may API-fill).
    """
    n = len(lats)
    elevations = np.zeros(n, dtype=np.float64)
    missing = np.zeros(n, dtype=bool)

    lat_floors = np.floor(lats).astype(int)
    lon_floors = np.floor(lons).astype(int)

    tile_keys = list(set(zip(lat_floors.tolist(), lon_floors.tolist())))
    for (tlat, tlon) in tile_keys:
        tile = _load_srtm_tile(int(tlat), int(tlon))
        mask = (lat_floors == tlat) & (lon_floors == tlon)
        if tile is None:
            missing[mask] = True
            continue
        frac_lat = lats[mask] - tlat
        frac_lon = lons[mask] - tlon
        rows = np.clip(((1.0 - frac_lat) * (_SRTM3_SAMPLES - 1)).astype(int), 0, _SRTM3_SAMPLES - 1)
        cols = np.clip((frac_lon * (_SRTM3_SAMPLES - 1)).astype(int), 0, _SRTM3_SAMPLES - 1)
        elevations[mask] = tile[rows, cols]

    return elevations, missing


def _open_elevation_lookup_coords(
    lats: np.ndarray,
    lons: np.ndarray,
    batch_size: int = 500,
) -> np.ndarray:
    """Query Open-Elevation for arbitrary coordinate arrays (same length)."""
    import requests

    n = len(lats)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out

    url = "https://api.open-elevation.com/api/v1/lookup"
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_lat = lats[start:end]
        batch_lon = lons[start:end]
        payload = {
            "locations": [
                {"latitude": float(lat), "longitude": float(lon)}
                for lat, lon in zip(batch_lat, batch_lon)
            ]
        }
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = requests.post(url, json=payload, timeout=90)
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                if len(results) != end - start:
                    raise RuntimeError("Open-Elevation response size mismatch")
                for i, r in enumerate(results):
                    out[start + i] = float(r["elevation"])
                break
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                if attempt == _MAX_RETRIES:
                    _log.warning(
                        "Open-Elevation gap-fill failed for batch %d–%d: %s (left as 0 m).",
                        start, end, exc,
                    )
                else:
                    time.sleep(_RETRY_BACKOFF_S * attempt)
    return out


def attach_node_elevation(
    G: nx.DiGraph,
    provider: ElevationProvider = "srtm",
    batch_size: int = 500,
) -> nx.DiGraph:
    """
    Attach ``elevation_m`` to every node in the movement graph.

    Parameters
    ----------
    provider
        ``"srtm"`` — SRTM3 tiles first (USGS download when needed). Any tile that
        cannot be loaded is filled via the Open-Elevation API so slopes stay
        realistic (small amount of HTTP only for missing tiles).

        ``"open_elevation"`` — public Open-Elevation API for every node.

        ``"none"`` — elevation = 0 everywhere (no network, no realism).
    batch_size
        Points per Open-Elevation POST (``open_elevation`` and SRTM gap-fill).
    """
    any_has = any("elevation_m" in G.nodes[n] for n in G.nodes())
    if any_has:
        return G

    if provider == "none":
        for n in G.nodes():
            G.nodes[n]["elevation_m"] = 0.0
        return G

    nodes = list(G.nodes())
    lats = np.array([float(G.nodes[n]["y"]) for n in nodes], dtype=np.float64)
    lons = np.array([float(G.nodes[n]["x"]) for n in nodes], dtype=np.float64)

    if provider == "srtm":
        _log.info("Looking up elevation for %d nodes from local SRTM3 tiles…", len(nodes))
        elevations, missing = _batch_srtm_elevations(lats, lons)
        if missing.any():
            n_miss = int(missing.sum())
            _log.info(
                "SRTM missing for %d / %d nodes — gap-filling via Open-Elevation…",
                n_miss, len(nodes),
            )
            api_z = _open_elevation_lookup_coords(
                lats[missing], lons[missing], batch_size=batch_size,
            )
            elevations[missing] = api_z
        for i, n in enumerate(nodes):
            G.nodes[n]["elevation_m"] = float(elevations[i])
        return G

    if provider == "open_elevation":
        return _attach_via_open_elevation(G, nodes, lats, lons, batch_size)

    raise ValueError(f"Unknown elevation provider: {provider!r}")


def _attach_via_open_elevation(
    G: nx.DiGraph,
    nodes: list,
    lats: np.ndarray,
    lons: np.ndarray,
    batch_size: int,
) -> nx.DiGraph:
    """Open-Elevation HTTP API for every node (with per-batch retries)."""
    n = len(nodes)
    total_batches = (n + batch_size - 1) // batch_size
    _log.info("Fetching elevation for %d nodes via API in %d batches…", n, total_batches)
    elevations = _open_elevation_lookup_coords(lats, lons, batch_size=batch_size)
    for i, node in enumerate(nodes):
        G.nodes[node]["elevation_m"] = float(elevations[i])
    return G


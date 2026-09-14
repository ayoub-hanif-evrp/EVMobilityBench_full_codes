"""
Load ``countries_data.json`` and resolve station-related fields for a (country, city).

Merge order:
  1. Top-level ``default`` block (baseline; used when country is unknown).
  2. Country entry, if ``country`` matches a key (case-insensitive).
  3. City entry under that country's ``cities``, if ``city`` matches (case-insensitive).

Keeping this logic in a dedicated module avoids bloating ``generator.py`` and makes
the merge rules testable without running OSM downloads.
"""

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional


def _find_matching_key(candidates: Any, wanted: str) -> Optional[str]:
    """Return the actual dict key whose casefold equals wanted.casefold(), or None."""
    if not wanted or not isinstance(candidates, dict):
        return None
    wf = wanted.strip().casefold()
    for k in candidates:
        if isinstance(k, str) and k.strip().casefold() == wf:
            return k
    return None


def _merge_station_profile(base: Dict[str, Any], overlay: Dict[str, Any]) -> None:
    """Merge overlay into base in place; skip ``cities``; deep-merge ``slot_defaults``."""
    for key, val in overlay.items():
        if key == "cities":
            continue
        if key == "slot_defaults" and isinstance(val, dict):
            existing = base.get("slot_defaults")
            if isinstance(existing, dict):
                merged = {**existing, **val}
                base["slot_defaults"] = merged
            else:
                base["slot_defaults"] = deepcopy(val)
        else:
            base[key] = deepcopy(val) if isinstance(val, dict) else val


@lru_cache(maxsize=1)
def load_countries_data() -> Dict[str, Any]:
    path = Path(__file__).resolve().with_name("countries_data.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_station_defaults(
    bundle: Dict[str, Any],
    country: str,
    city: str,
) -> Dict[str, Any]:
    """
    Effective dict of station pricing / slots / carbon fields for this place.

    Parameters
    ----------
    bundle
        Parsed ``countries_data.json`` (must contain a ``default`` key).
    country, city
        From ``GenerationConfig``; matching is case-insensitive.
    """
    if "default" not in bundle or not isinstance(bundle["default"], dict):
        raise ValueError("countries_data.json must define a top-level 'default' object.")

    result: Dict[str, Any] = deepcopy(bundle["default"])

    country_key = _find_matching_key(
        {k: None for k in bundle if k != "default"},
        country,
    )
    if country_key is None:
        return result

    country_block = bundle.get(country_key)
    if not isinstance(country_block, dict):
        return result

    _merge_station_profile(result, country_block)

    cities = country_block.get("cities")
    if not isinstance(cities, dict):
        return result

    city_key = _find_matching_key(cities, city)
    if city_key is None:
        return result

    city_block = cities.get(city_key)
    if isinstance(city_block, dict):
        _merge_station_profile(result, city_block)

    return result

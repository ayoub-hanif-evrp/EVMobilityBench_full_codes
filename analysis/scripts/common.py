"""Minimal graph/config I/O for figure regeneration."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from paths import CAMPAIGN_JSON, CITY_MANIFEST, GRAPHS_CACHE


def load_city_manifest() -> list[dict]:
    return json.loads(CITY_MANIFEST.read_text(encoding="utf-8"))["cities"]


def city_country_map() -> dict[str, str]:
    return {r["city"]: r["country"] for r in load_city_manifest()}


def graph_path(city: str, country: str) -> Path:
    safe = f"{city}_{country}".replace(" ", "_")
    return GRAPHS_CACHE / f"{safe}.pkl"


def load_graph(city: str, country: str) -> Any:
    path = graph_path(city, country)
    with path.open("rb") as f:
        return pickle.load(f)


def load_campaign() -> dict:
    return json.loads(CAMPAIGN_JSON.read_text(encoding="utf-8"))

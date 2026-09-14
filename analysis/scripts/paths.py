"""Shared paths for the analysis package."""
from __future__ import annotations

from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPTS_DIR.parent
REPO_ROOT = ANALYSIS_DIR.parent
CONFIG_DIR = ANALYSIS_DIR / "config"
RESULTS_DIR = ANALYSIS_DIR / "results"
DATA_DIR = RESULTS_DIR / "data"
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_DIR = ANALYSIS_DIR / "cache"
GRAPHS_CACHE = CACHE_DIR / "graphs"
CAMPAIGN_JSON = CONFIG_DIR / "campaign.json"
CITY_MANIFEST = CONFIG_DIR / "city_manifest.json"
DEPOT_JSON = CONFIG_DIR / "depot_facilities.json"
FRAMEWORK_SRC = REPO_ROOT / "src" / "evrp_instance_generator_framework"

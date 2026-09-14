"""Simple customer CSV import helpers shared by notebooks and web app."""

from __future__ import annotations

import csv
import io
from dataclasses import replace
from pathlib import Path
from typing import BinaryIO, Iterable, TextIO

from ..exceptions import EvrpUserError
from ..types import CustomerRecord, PipelineState

REQUIRED_CUSTOMER_COLUMNS = (
    "id",
    "lat",
    "lon",
    "movement_node_id",
    "snap_distance_m",
    "demand",
    "service_time_s",
    "parking_time_s",
    "time_open_s",
    "time_close_s",
)


def _rows_from_dict_reader(reader: csv.DictReader) -> Iterable[dict]:
    if reader.fieldnames is None:
        raise EvrpUserError("Customer CSV has no header row.")
    missing = [c for c in REQUIRED_CUSTOMER_COLUMNS if c not in reader.fieldnames]
    if missing:
        raise EvrpUserError(
            "Customer CSV is missing required columns: " + ", ".join(missing)
        )
    return reader


def _to_customer_record(row: dict, row_index: int) -> CustomerRecord:
    try:
        return CustomerRecord(
            id=int(row["id"]),
            lat=float(row["lat"]),
            lon=float(row["lon"]),
            movement_node_id=int(row["movement_node_id"]),
            snap_distance_m=float(row["snap_distance_m"]),
            demand=int(row["demand"]),
            service_time_s=int(row["service_time_s"]),
            parking_time_s=int(row["parking_time_s"]),
            time_open_s=int(row["time_open_s"]),
            time_close_s=int(row["time_close_s"]),
        )
    except Exception as exc:  # noqa: BLE001
        raise EvrpUserError(
            f"Invalid value in customer CSV at row {row_index}: {exc}"
        ) from exc


def load_customers_from_csv(
    csv_source: str | Path | bytes | BinaryIO | TextIO,
) -> list[CustomerRecord]:
    """
    Load customer records from CSV.

    Accepted sources:
    - filesystem path (str/Path)
    - raw CSV bytes
    - opened binary/text file object
    """
    if isinstance(csv_source, (str, Path)):
        with Path(csv_source).open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = _rows_from_dict_reader(reader)
            customers = [_to_customer_record(r, i) for i, r in enumerate(rows, start=2)]
    elif isinstance(csv_source, bytes):
        f = io.StringIO(csv_source.decode("utf-8"))
        reader = csv.DictReader(f)
        rows = _rows_from_dict_reader(reader)
        customers = [_to_customer_record(r, i) for i, r in enumerate(rows, start=2)]
    else:
        if hasattr(csv_source, "read"):
            raw = csv_source.read()
        else:
            raise EvrpUserError("Unsupported CSV source type.")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        reader = csv.DictReader(io.StringIO(raw))
        rows = _rows_from_dict_reader(reader)
        customers = [_to_customer_record(r, i) for i, r in enumerate(rows, start=2)]

    if not customers:
        raise EvrpUserError("Customer CSV is empty.")
    return customers


def apply_customers_to_state(state: PipelineState, customers: list[CustomerRecord]) -> PipelineState:
    """Replace generated customers with imported ones and sync config count."""
    if not customers:
        raise EvrpUserError("Imported customers list is empty.")
    state.customers = list(customers)
    state.config = replace(state.config, num_customers=len(customers))
    return state


def resolve_num_customers_from_config(config) -> int:
    """
    Resolve customer count from config.

    Priority:
    1) explicit config.num_customers
    2) len(customer_csv_path content)
    """
    if getattr(config, "num_customers", None) is not None:
        return int(config.num_customers)
    csv_path = getattr(config, "customer_csv_path", None)
    if csv_path:
        return len(load_customers_from_csv(csv_path))
    raise EvrpUserError("Provide either num_customers or customer_csv_path in GenerationConfig.")

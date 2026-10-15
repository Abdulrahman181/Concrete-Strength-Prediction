"""Reusable data-contract helpers for the concrete strength workflow."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42
TARGET = "csMPa"
FEATURES = [
    "cement",
    "slag",
    "flyash",
    "water",
    "superplasticizer",
    "coarseaggregate",
    "fineaggregate",
    "age",
]
REQUIRED_COLUMNS = FEATURES + [TARGET]
MIN_ROWS_FOR_HOLDOUT_METRICS = 6


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the expected Yeh concrete-strength CSV.

    The returned frame contains only the required columns, in a stable order,
    with numeric finite values, no missing rows, and duplicate rows removed.
    At least six rows are required so a 20% holdout contains two observations
    and the reported R² metric is defined.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place Concrete_Data_Yeh.csv in "
            "data/ or set CONCRETE_DATA_PATH to an existing CSV file."
        )

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next((row for row in csv.reader(handle) if row), None)
        if header is None:
            raise ValueError(f"Dataset at {path} is empty.")
        if len(header) != len(set(header)):
            duplicates = sorted({column for column in header if header.count(column) > 1})
            raise ValueError(f"Dataset contains duplicate column names: {duplicates}")
        raw = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset at {path} is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Dataset at {path} is not a valid CSV file: {exc}") from exc

    if raw.columns.duplicated().any():
        duplicates = raw.columns[raw.columns.duplicated()].tolist()
        raise ValueError(f"Dataset contains duplicate column names: {duplicates}")

    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    frame = raw[REQUIRED_COLUMNS].copy()
    invalid_columns: list[str] = []
    for column in REQUIRED_COLUMNS:
        original = frame[column]
        converted = pd.to_numeric(original, errors="coerce")
        if (original.notna() & converted.isna()).any():
            invalid_columns.append(column)
        frame[column] = converted
    if invalid_columns:
        raise ValueError(f"Dataset contains non-numeric values in: {invalid_columns}")

    null_columns = frame.columns[frame.isna().any()].tolist()
    if null_columns:
        raise ValueError(f"Dataset contains missing values in: {null_columns}")

    non_finite_columns = [
        column for column in REQUIRED_COLUMNS if not np.isfinite(frame[column].to_numpy()).all()
    ]
    if non_finite_columns:
        raise ValueError(f"Dataset contains non-finite values in: {non_finite_columns}")

    frame = frame.drop_duplicates().reset_index(drop=True)
    if frame.empty:
        raise ValueError("Dataset has no rows after loading and deduplication.")
    if len(frame) < MIN_ROWS_FOR_HOLDOUT_METRICS:
        raise ValueError(
            f"Dataset needs at least {MIN_ROWS_FOR_HOLDOUT_METRICS} unique rows for "
            f"the 20% holdout metrics; found {len(frame)}."
        )
    return frame

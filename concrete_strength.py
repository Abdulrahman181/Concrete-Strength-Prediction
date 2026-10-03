"""Leakage-aware utilities for the concrete-strength regression example.

The estimator and metrics are educational research outputs, not engineering
advice or a validated basis for structural design.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

FEATURE_COLUMNS = (
    "cement",
    "slag",
    "flyash",
    "water",
    "superplasticizer",
    "coarseaggregate",
    "fineaggregate",
    "age",
)
TARGET_COLUMN = "csMPa"
REQUIRED_COLUMNS = (*FEATURE_COLUMNS, TARGET_COLUMN)
RANDOM_STATE = 42
SPLIT_STRATEGY = "grouped recipe split: 60% train, 20% validation, 20% test (approximate)"


@dataclass(frozen=True)
class DataSplit:
    """Data partitions plus composition identifiers used to prevent overlap."""

    x_train: pd.DataFrame
    x_validation: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    groups_train: np.ndarray
    groups_validation: np.ndarray
    groups_test: np.ndarray


@dataclass(frozen=True)
class Evaluation:
    """Validation selection results and final untouched-test results."""

    model_name: str
    model: Any
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    candidate_validation_metrics: dict[str, dict[str, float]]


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and return only the approved numeric model columns.

    Additional input columns are deliberately ignored, so identifiers or an
    accidental target-derived column can never silently become predictors.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Dataset must be a pandas DataFrame.")
    if not frame.columns.is_unique:
        raise ValueError("Dataset contains duplicate column names.")

    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
    if len(frame) < 8:
        raise ValueError("Dataset must contain at least 8 rows for train/validation/test splits.")

    data = frame.loc[:, REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        try:
            data[column] = pd.to_numeric(data[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Column {column!r} must contain numeric values.") from exc

    values = data.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Dataset contains missing or non-finite numeric values.")
    if (data.loc[:, FEATURE_COLUMNS] < 0).any().any():
        raise ValueError("Concrete mixture features must be non-negative.")
    if (data["age"] <= 0).any():
        raise ValueError("Concrete age must be greater than zero days.")
    if (data[TARGET_COLUMN] <= 0).any():
        raise ValueError("Concrete strength values must be greater than zero MPa.")
    if data.loc[:, FEATURE_COLUMNS].drop_duplicates().shape[0] < 8:
        raise ValueError(
            "Dataset must contain at least 8 unique concrete recipes for grouped splitting."
        )
    return data


def _recipe_groups(data: pd.DataFrame) -> np.ndarray:
    """Create stable integer group IDs for distinct feature recipes."""
    groups, _ = pd.factorize(pd.MultiIndex.from_frame(data.loc[:, FEATURE_COLUMNS]), sort=True)
    return groups


def split_data(frame: pd.DataFrame, random_state: int = RANDOM_STATE) -> DataSplit:
    """Split by recipe so identical feature vectors cannot cross partitions."""
    if not isinstance(random_state, int) or not 0 <= random_state < 2**32:
        raise ValueError("random_state must be an integer between 0 and 2**32 - 1.")
    data = validate_dataset(frame)
    groups = _recipe_groups(data)
    indices = np.arange(len(data))

    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
    train_validation_idx, test_idx = next(outer.split(indices, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
    train_idx, validation_idx = next(
        inner.split(train_validation_idx, groups=groups[train_validation_idx])
    )
    train_idx = train_validation_idx[train_idx]
    validation_idx = train_validation_idx[validation_idx]

    def features(idx: np.ndarray) -> pd.DataFrame:
        return data.iloc[idx].loc[:, FEATURE_COLUMNS].reset_index(drop=True)

    def target(idx: np.ndarray) -> pd.Series:
        return data.iloc[idx][TARGET_COLUMN].reset_index(drop=True)

    split = DataSplit(
        x_train=features(train_idx),
        x_validation=features(validation_idx),
        x_test=features(test_idx),
        y_train=target(train_idx),
        y_validation=target(validation_idx),
        y_test=target(test_idx),
        groups_train=groups[train_idx],
        groups_validation=groups[validation_idx],
        groups_test=groups[test_idx],
    )
    group_sets = [set(split.groups_train), set(split.groups_validation), set(split.groups_test)]
    if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Grouped split failed: a recipe appears in more than one partition.")
    if min(len(partition) for partition in (split.y_train, split.y_validation, split.y_test)) < 2:
        raise ValueError("Dataset is too small to place at least two rows in every split.")
    return split


def _metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def _new_models(random_state: int) -> dict[str, Any]:
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=1,
        ),
    }


def evaluate_models(frame: pd.DataFrame, random_state: int = RANDOM_STATE) -> Evaluation:
    """Select by validation MAE, refit on train+validation, and test once.

    Identical concrete recipes are assigned to a single partition. The test
    set is untouched until model selection is complete.
    """
    data = validate_dataset(frame)
    split = split_data(data, random_state=random_state)
    models = _new_models(random_state)
    validation_results: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        model.fit(split.x_train, split.y_train)
        validation_results[name] = _metrics(
            split.y_validation, model.predict(split.x_validation)
        )

    # Stable tie-breaking follows insertion order; no test-set metric is used.
    selected_name = min(models, key=lambda name: validation_results[name]["mae"])
    selected_model = _new_models(random_state)[selected_name]
    train_validation_x = pd.concat(
        [split.x_train, split.x_validation], ignore_index=True
    )
    train_validation_y = pd.concat(
        [split.y_train, split.y_validation], ignore_index=True
    )
    selected_model.fit(train_validation_x, train_validation_y)
    test_results = _metrics(split.y_test, selected_model.predict(split.x_test))
    return Evaluation(
        model_name=selected_name,
        model=selected_model,
        validation_metrics=validation_results[selected_name],
        test_metrics=test_results,
        candidate_validation_metrics=validation_results,
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def save_model_artifact(
    evaluation: Evaluation, artifact_dir: str | Path, random_state: int = RANDOM_STATE
) -> tuple[Path, Path]:
    """Explicitly export the fitted estimator and a data-free metadata file."""
    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "model.pkl"
    metadata_path = directory / "model_metadata.json"

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=directory, delete=False) as stream:
            temporary_path = Path(stream.name)
            pickle.dump(evaluation.model, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        digest = hashlib.sha256(temporary_path.read_bytes()).hexdigest()
        os.replace(temporary_path, model_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    metadata = {
        "format_version": 1,
        "model_file": model_path.name,
        "model_sha256": digest,
        "model_name": evaluation.model_name,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "random_state": random_state,
        "split_strategy": SPLIT_STRATEGY,
        "selection_metric": "validation_mae",
    }
    _write_json_atomic(metadata_path, metadata)
    return model_path, metadata_path


def _default_dataset_path() -> Path | None:
    repository_root = Path(__file__).resolve().parent
    for candidate in (
        repository_root / "data" / "Concrete_Data_Yeh.csv",
        repository_root / "Concrete_Data_Yeh.csv",
    ):
        if candidate.is_file():
            return candidate
    return None


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the requested CSV, without exposing absolute paths in errors."""
    selected_path = Path(path) if path is not None else _default_dataset_path()
    if selected_path is None or not selected_path.is_file():
        raise FileNotFoundError(
            "Dataset not found. Supply --data PATH or place Concrete_Data_Yeh.csv "
            "in the repository root or data/ directory."
        )
    try:
        return validate_dataset(pd.read_csv(selected_path))
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(
            "Dataset CSV could not be read; check its encoding and CSV format."
        ) from exc


def _report(evaluation: Evaluation, random_state: int) -> dict[str, Any]:
    return {
        "selected_model": evaluation.model_name,
        "selection_metric": "validation_mae",
        "candidate_validation_metrics": evaluation.candidate_validation_metrics,
        "selected_validation_metrics": evaluation.validation_metrics,
        "untouched_test_metrics": evaluation.test_metrics,
        "split_strategy": SPLIT_STRATEGY,
        "random_state": random_state,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate an educational concrete-strength regressor."
    )
    parser.add_argument("--data", type=Path, help="Input CSV (never copied into the output).")
    parser.add_argument(
        "--random-state", type=int, default=RANDOM_STATE,
        help=f"Deterministic split/model seed (default: {RANDOM_STATE}).",
    )
    parser.add_argument(
        "--artifact-dir", type=Path,
        help="Opt in to writing model.pkl and data-free model_metadata.json.",
    )
    args = parser.parse_args(argv)

    try:
        data = load_dataset(args.data)
        evaluation = evaluate_models(data, random_state=args.random_state)
        print(
            json.dumps(
                _report(evaluation, args.random_state),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        if args.artifact_dir is not None:
            model_path, metadata_path = save_model_artifact(
                evaluation, args.artifact_dir, random_state=args.random_state
            )
            print(f"Model artifact written: {model_path.name}")
            print(f"Metadata written: {metadata_path.name}")
    except (FileNotFoundError, TypeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

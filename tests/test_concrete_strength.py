import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from concrete_strength import (
    FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    evaluate_models,
    load_dataset,
    save_model_artifact,
    split_data,
    validate_dataset,
)


def sample_data(recipes: int = 60, repeats: int = 2) -> pd.DataFrame:
    rows = []
    for recipe in range(recipes):
        features = {
            "cement": 100 + recipe * 2,
            "slag": 10 + (recipe % 13),
            "flyash": 5 + (recipe % 7),
            "water": 130 + (recipe % 19),
            "superplasticizer": recipe % 11,
            "coarseaggregate": 800 + (recipe % 31),
            "fineaggregate": 650 + (recipe % 23),
            "age": 7 + (recipe % 365),
        }
        strength = 8.0 + 0.05 * features["cement"] + 0.02 * features["age"]
        for _ in range(repeats):
            rows.append({**features, "csMPa": strength})
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def test_validation_selects_expected_columns_and_rejects_bad_values():
    data = sample_data().assign(row_id=np.arange(len(sample_data())))
    validated = validate_dataset(data)
    assert tuple(validated.columns) == REQUIRED_COLUMNS

    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(data.drop(columns="csMPa"))

    invalid = sample_data()
    invalid.loc[0, "water"] = -1
    with pytest.raises(ValueError, match="non-negative"):
        validate_dataset(invalid)

    invalid = sample_data()
    invalid["cement"] = invalid["cement"].astype(float)
    invalid.loc[0, "cement"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        validate_dataset(invalid)


def test_recipe_groups_do_not_leak_between_partitions_and_seed_is_repeatable():
    data = sample_data()
    first = split_data(data, random_state=17)
    second = split_data(data, random_state=17)

    groups = [set(first.groups_train), set(first.groups_validation), set(first.groups_test)]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert first.x_train.equals(second.x_train)
    assert first.x_validation.equals(second.x_validation)
    assert first.x_test.equals(second.x_test)
    assert set(first.x_train.columns) == set(FEATURE_COLUMNS)


def test_evaluation_selects_on_validation_then_reports_test_metrics():
    data = sample_data()
    first = evaluate_models(data, random_state=17)
    second = evaluate_models(data, random_state=17)

    assert first.model_name in {"linear_regression", "random_forest"}
    assert set(first.candidate_validation_metrics) == {"linear_regression", "random_forest"}
    assert set(first.test_metrics) == {"mae", "rmse", "r2"}
    assert first.test_metrics == second.test_metrics
    assert first.validation_metrics == first.candidate_validation_metrics[first.model_name]


def test_model_artifact_metadata_is_data_free_and_hash_matches(tmp_path):
    result = evaluate_models(sample_data(), random_state=3)
    model_path, metadata_path = save_model_artifact(result, tmp_path, random_state=3)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["model_sha256"] == hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert metadata["feature_columns"] == list(FEATURE_COLUMNS)
    assert metadata["target_column"] == "csMPa"
    assert "row_count" not in metadata
    assert "dataset_path" not in metadata
    assert model_path.name == "model.pkl"


def test_missing_dataset_error_does_not_echo_sensitive_path(tmp_path):
    private_path = tmp_path / "private" / "measurement.csv"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_dataset(private_path)
    assert str(private_path) not in str(exc_info.value)

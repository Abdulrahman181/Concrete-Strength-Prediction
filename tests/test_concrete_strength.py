from __future__ import annotations

import pandas as pd
import pytest

from concrete_strength import FEATURES, TARGET, load_dataset


def valid_frame(rows: int = 6) -> pd.DataFrame:
    values = {column: [float(index + 1) for index in range(rows)] for column in FEATURES}
    values[TARGET] = [float(index + 10) for index in range(rows)]
    return pd.DataFrame(values)


def test_load_dataset_keeps_schema_and_removes_duplicates(tmp_path):
    frame = valid_frame()
    path = tmp_path / "concrete.csv"
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(path, index=False)

    loaded = load_dataset(path)

    assert loaded.columns.tolist() == FEATURES + [TARGET]
    assert len(loaded) == len(frame)
    assert loaded.isna().sum().sum() == 0


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("cement", "not-a-number", "non-numeric"),
        ("water", "", "missing"),
        ("age", "inf", "non-finite"),
    ],
)
def test_load_dataset_reports_invalid_values(tmp_path, column, value, message):
    frame = valid_frame()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    path = tmp_path / "invalid.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        load_dataset(path)


def test_load_dataset_rejects_missing_columns_and_too_few_unique_rows(tmp_path):
    missing = valid_frame().drop(columns=[TARGET])
    missing_path = tmp_path / "missing.csv"
    missing.to_csv(missing_path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(missing_path)

    short_path = tmp_path / "short.csv"
    valid_frame(rows=2).to_csv(short_path, index=False)
    with pytest.raises(ValueError, match="at least 6 unique rows"):
        load_dataset(short_path)


def test_load_dataset_rejects_duplicate_csv_headers(tmp_path):
    frame = valid_frame()
    header = frame.columns.tolist() + [FEATURES[0]]
    row = frame.iloc[0].tolist() + [frame.iloc[0][FEATURES[0]]]
    path = tmp_path / "duplicate-header.csv"
    path.write_text(",".join(header) + "\n" + ",".join(map(str, row)) + "\n")

    with pytest.raises(ValueError, match="duplicate column names"):
        load_dataset(path)

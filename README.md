# Concrete Strength Prediction

A small, reproducible scikit-learn workflow for predicting concrete compressive strength (`csMPa`) from the Yeh concrete dataset. The repository currently contains the cleaned notebook and validation tooling; **the input dataset is not included**.

## Dataset

Place the CSV at `data/Concrete_Data_Yeh.csv`, or point the notebook at another file with the `CONCRETE_DATA_PATH` environment variable:

```bash
export CONCRETE_DATA_PATH=/absolute/path/to/Concrete_Data_Yeh.csv
```

The CSV must contain these columns and at least six unique, finite rows:

`cement`, `slag`, `flyash`, `water`, `superplasticizer`, `coarseaggregate`, `fineaggregate`, `age`, and `csMPa`.

The notebook fails early with an actionable error when the file is missing, required columns are absent, values are non-numeric or non-finite, missing values remain after loading, or the dataset is too small for a two-row test holdout. Duplicate rows are removed before this minimum-size check. See [`data/README.md`](data/README.md) for the same contract.

## Workflow and safeguards

- Duplicate rows are removed before splitting and the resulting index is reset.
- A fixed `random_state=42` makes the train/test split and random forest reproducible.
- The test set is held out with `train_test_split(test_size=0.2)`.
- Scaling is inside a scikit-learn `Pipeline`, so it is fitted only on training data; no transformation is fitted on the full dataset.
- Metrics are calculated on the held-out test set using R², MAE, and RMSE. A single split is not a substitute for cross-validation.
- Notebook outputs are intentionally cleared so committed results cannot be mistaken for a fresh run.

## Run

Use Python 3.9 or newer. From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python validate_repo.py
pytest -q
jupyter notebook "Concrete Strength Prediction.ipynb"
```

`validate_repo.py` checks Python and notebook syntax, rejects stale notebook outputs, and detects Windows-specific paths. Ruff can also be run with:

```bash
ruff check .
```

The notebook cannot complete until a compatible dataset is supplied. This repository does not claim model scores without that input.

## Project files

- `Concrete Strength Prediction.ipynb` — portable EDA and deterministic baseline comparison (linear regression and random forest).
- `concrete_strength.py` — reusable dataset schema and value validation used by the notebook.
- `tests/test_concrete_strength.py` — regression tests for loading and validation edge cases.
- `requirements.txt` — runtime and validation dependencies.
- `validate_repo.py` — dependency-light static validation.
- `data/README.md` — input data contract.
- `ENGINEERING_REVIEW.md` — audit findings, fixes, and validation evidence.

## License and attribution

The original project and dataset attribution should be confirmed by the repository owner before redistribution. No dataset is bundled here.

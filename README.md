# Concrete Strength Prediction

An educational regression example for estimating the compressive strength (`csMPa`) of concrete mixtures from the Yeh dataset schema. The repository contains an exploratory notebook and a small Python training module. It does **not** include a dataset, trained model, verified data source, benchmark result, or license grant.

> **Research/education only.** This project is not validated or certified for structural design, construction decisions, material acceptance, or other safety-critical use. Model output is not engineering advice and must not replace applicable standards, laboratory testing, or a qualified professional.

## Data

Obtain a CSV you are authorized to use and place it at `data/Concrete_Data_Yeh.csv` or `Concrete_Data_Yeh.csv` in the repository root. You may also pass an explicit path with `--data`. The repository does not endorse or verify a download source or dataset license.

The input must contain numeric columns `cement`, `slag`, `flyash`, `water`, `superplasticizer`, `coarseaggregate`, `fineaggregate`, `age`, and `csMPa`. Features must be finite and non-negative; age and strength must be positive. Additional columns are ignored, so identifiers or unrelated measurements cannot become model inputs. The training utility requires at least eight distinct feature recipes to form grouped partitions. See [`schemas/dataset.schema.json`](schemas/dataset.schema.json) for the machine-readable column contract.

## Setup

Python 3.10–3.12 is supported. Install the project and optional notebook/test tools in a virtual environment:

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -e '.[notebook,test]'
```

Run the notebook from the repository root:

```bash
jupyter notebook "Concrete Strength Prediction.ipynb"
```

Or run the reproducible command-line workflow:

```bash
python -m concrete_strength --data data/Concrete_Data_Yeh.csv
```

If the file is in one of the default repository locations, `--data` can be omitted. Errors avoid printing absolute input paths. A fixed default seed (`42`) is used; supply `--random-state` to change it. Results can still vary with dataset contents, dependency versions, and platform.

## Evaluation and leakage controls

The approved eight mixture features are predictors and `csMPa` is the target. Exact duplicate feature recipes are grouped into a single partition, so the same recipe cannot appear in both fitting and evaluation data. A deterministic grouped split targets approximately 60% train, 20% validation, and 20% test (actual proportions depend on the number and distribution of unique recipes).

Linear regression and a seeded, single-worker random forest are fit on training rows. The model is selected by **validation MAE**; the selected model is then refit on train plus validation rows and scored once on the untouched test partition. The test score is not used for model choice or tuning. Printed metrics are aggregate MAE, RMSE, and R² only; no row-level predictions or input rows are written to disk by default. Do not treat a single holdout score as external validation or a performance guarantee.

To run checks locally:

```bash
python -m pytest
ruff check .
python -m compileall -q concrete_strength.py tests
```

GitHub Actions runs these checks on pull requests and pushes for Python 3.10, 3.11, and 3.12. The notebook is committed without saved cell outputs.

## Optional model export and privacy

By default, no model file or metrics file is created. To explicitly export the fitted model and its data-free schema/version/seed metadata:

```bash
python -m concrete_strength --data data/Concrete_Data_Yeh.csv --artifact-dir artifacts
```

The metadata contract is [`schemas/model-metadata.schema.json`](schemas/model-metadata.schema.json); the metadata contains no dataset path, row-level data, or performance metrics. The serialized model can nevertheless reveal information about its training data. Keep it private when appropriate. **Never load a model pickle from an untrusted source**: Python pickle deserialization can execute code. No model-loading command is provided by this project.

The input dataset is never copied into the repository or artifact directory by the code. Notebook displays and plots may still reveal characteristics of whatever dataset you load; review and clear any outputs before sharing a notebook, and do not commit private source data.

## Project files

- `concrete_strength.py` — validation, recipe-grouped splitting, train/validation/test workflow, CLI, and opt-in artifact export.
- `Concrete Strength Prediction.ipynb` — exploratory summaries/plots and use of the shared evaluation module.
- `schemas/` — input and model-metadata schemas.
- `tests/` — synthetic-data tests (no actual dataset or performance claim).
- `.github/workflows/ci.yml` — automated checks.

## Author

Abdul Rahman Ahmed

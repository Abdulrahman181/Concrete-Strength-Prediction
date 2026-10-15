# Engineering Review — Concrete Strength Prediction

**Repository:** `Abdulrahman181/Concrete-Strength-Prediction`  
**Review scope:** Python, Jupyter notebook, README, dependency/configuration files, data-path handling, reproducibility, split/leakage safeguards, and static validation.  
**Review status:** Repaired locally; no changes were pushed to GitHub.

## Executive summary

The repository originally contained one notebook and a README, but no tracked dataset, dependency lock/list, ignore rules, or automated validation. The notebook could not be run portably because it used an absolute Windows path to a local CSV. It also attempted to scale `X` before `X` was defined, contained duplicate imports, retained stale outputs, and fit preprocessing before the train/test split. The README described models and scores that were not a reliable, reproducible contract for the checked-in code.

The notebook is now a clean, output-free, portable baseline workflow. It validates the input schema, removes duplicates before splitting, uses a fixed random seed, and keeps scaling inside a scikit-learn pipeline fitted only on training data. It compares linear regression and random forest with R², MAE, and RMSE on one held-out test set. No performance result is asserted in this review because the dataset is not present.

## Findings and severity

### Critical / blocking

1. **Dataset was absent from the repository and the notebook used a machine-specific path.** The original loader referenced `C:\Users\...\Concrete_Data_Yeh.csv`, which cannot work on another machine or in this Sandbox. The notebook now defaults to `data/Concrete_Data_Yeh.csv` and supports `CONCRETE_DATA_PATH`. Missing files produce an actionable `FileNotFoundError`; missing columns, non-numeric values, empty data, and nulls produce explicit validation errors.
2. **Notebook execution order was broken.** `X_scaled = scaler.fit_transform(X)` appeared before `X` was assigned. The rewritten notebook defines the feature matrix after loading and performs transformations through a pipeline after the split.
3. **Potential preprocessing leakage.** The original scaler was fitted before the train/test split. The repaired linear-regression pipeline learns scaling parameters only from `X_train`; the tree model does not require scaling.

### High

4. **No declared dependencies or static validation existed.** Added `requirements.txt`, `pyproject.toml` Ruff configuration, and dependency-light `validate_repo.py`.
5. **Committed notebook outputs and execution counts were stale.** All code-cell outputs are removed and execution counts are reset to `null`, preventing old metrics/plots from being mistaken for a fresh run.
6. **Results were based on a single holdout split.** The workflow now documents that R²/MAE/RMSE are holdout metrics and not cross-validation estimates. Cross-validation remains an appropriate future improvement for model selection.

### Medium

7. **Duplicate and inconsistent imports existed.** Duplicate `RandomForestRegressor` and `XGBRegressor` imports were removed; the repaired notebook has one sorted import block.
8. **README claims did not match a reproducible implementation.** It listed classification and clustering models not present in the notebook and asserted scores without a current, runnable data contract. README was rewritten to describe the actual baseline workflow and explicitly avoid unsupported scores.
9. **No data contract was documented.** Added `data/README.md` with the required nine columns and the intentional absence of the source CSV.

## Changes made

- Replaced `Concrete Strength Prediction.ipynb` with a portable, deterministic, output-free notebook.
- Added schema and value checks for the expected Yeh concrete-strength CSV.
- Added duplicate removal before splitting, fixed `random_state=42`, and a 20% holdout split.
- Added a `StandardScaler` + `LinearRegression` pipeline and a seeded, parallel `RandomForestRegressor` baseline.
- Added held-out R², MAE, and RMSE calculations without hard-coded results.
- Added `requirements.txt` for runtime and validation dependencies.
- Added `.gitignore` to keep local datasets, environments, caches, and generated outputs out of Git.
- Added `pyproject.toml` Ruff settings.
- Added `validate_repo.py` to parse Python and notebook code, reject non-portable paths, and reject stale notebook outputs.
- Added `data/README.md` documenting the required input path/schema.
- Rewrote `README.md` with accurate setup, usage, safeguards, and limitations.
- Added this review report.

## Validation evidence

The following checks passed locally:

| Check | Result | Notes |
|---|---|---|
| `python3 validate_repo.py` | Passed | Python/notebook AST syntax, path portability, and empty code outputs |
| `ruff check . --output-format concise` | Passed | Ruff reported `All checks passed!` |
| `python3 -m compileall -q .` | Passed | Python source compilation completed successfully |
| Notebook JSON inspection | Passed | 10 cells; all 7 code cells have `execution_count: null` and zero outputs |
| Git push | Not performed | Changes are local only, as requested |

## Not run / caveats

- The repository has no dataset, so the end-to-end notebook was **not** run and no model score is claimed here.
- The Sandbox did not have `scikit-learn`, `xgboost`, `nbformat`, or `nbconvert` importable during the audit. Ruff and the standard-library validation/compile checks were available and passed. Installing `requirements.txt` is required before opening/executing the notebook in a complete environment.
- The repaired notebook intentionally uses only the core dependencies declared in `requirements.txt`; it does not retain an optional XGBoost cell whose dependency was undeclared and unavailable.
- The single holdout split is suitable as a deterministic baseline, not as a final model-selection protocol. Add cross-validation and an untouched final test protocol before using results for a production or scientific claim.

## Second Senior Review

### Scope and findings

This second pass re-read every tracked Python, notebook, configuration, dependency,
README, data-contract, and review file; inspected the repair history (`2d33ca7` to
`fa34c8f`); and re-ran static checks and tests. The repaired notebook had no remaining
syntax errors, stale outputs, execution counts, Windows paths, or obvious train/test
leakage. The remaining material issues were:

1. **The loader was notebook-only and not unit-testable.** Its data contract could not
   be tested independently, and the project had no tests; `pytest -q` therefore found
   no tests.
2. **Malformed values were not fully guarded.** The previous loader did not explicitly
   reject non-finite values such as `inf`, did not give a deliberate empty-CSV error,
   and accepted too few rows for a meaningful two-observation test holdout/R².
3. **CSV schema edge cases were under-specified.** Duplicate header names could be
   silently mangled by pandas, and the flat-layout project lacked pytest import
   configuration when tests were added.

### Repairs made

- Added `concrete_strength.py` as the reusable loader and schema contract used by the
  notebook. It now handles missing/empty/parser-error files, duplicate headers,
  missing columns, non-numeric values, missing values, non-finite values, duplicate
  rows, and the six-unique-row minimum needed by the documented holdout metrics.
- Refactored notebook cells to import that loader and constants, while retaining a
  deterministic split (`random_state=42`) and training-only pipeline scaling.
- Added six focused tests in `tests/test_concrete_strength.py` covering valid loading,
  deduplication, invalid values, missing columns, too-small datasets, and duplicate
  headers; added `pytest` and flat-layout `pythonpath`/`testpaths` configuration.
- Updated README and `data/README.md` to document the stricter contract and test
  command. Notebook outputs and execution counts remain cleared.

### Second-pass validation

| Check | Result |
|---|---|
| `python3 validate_repo.py` | Passed |
| `ruff check . --output-format concise` | Passed |
| `python3 -m compileall -q .` | Passed |
| `pytest -q` | Passed — 6 tests |
| Notebook JSON/AST/output/path hygiene | Passed |
| End-to-end model execution | Not run — source dataset is intentionally absent |

### Remaining risks and non-claims

- No model scores or end-to-end execution are claimed because `data/Concrete_Data_Yeh.csv`
  is not checked in and no substitute dataset was invented.
- The workflow still uses one deterministic holdout and has no persisted model or
  prediction API; cross-validation, an untouched final test set, and artifact-serving
  code remain future work.
- Dependency ranges are minimum-version declarations rather than a fully locked
  environment, so reproducibility across future dependency releases is not absolute.

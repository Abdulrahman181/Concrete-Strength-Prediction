# Data directory

The dataset is intentionally not committed. Place the input CSV here as
`Concrete_Data_Yeh.csv`, or set the `CONCRETE_DATA_PATH` environment variable to
an existing CSV path before running the notebook.

The CSV must contain these columns: `cement`, `slag`, `flyash`, `water`,
`superplasticizer`, `coarseaggregate`, `fineaggregate`, `age`, and `csMPa`.
The loader validates the columns, rejects duplicate column names, converts and
checks finite numeric values, rejects missing values, and removes duplicate rows
before training. At least six unique rows are required so the notebook's 20%
holdout contains two observations and its R² metric is defined.

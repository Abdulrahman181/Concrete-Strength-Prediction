# Data directory

The dataset is intentionally not committed. Place the input CSV here as
`Concrete_Data_Yeh.csv`, or set the `CONCRETE_DATA_PATH` environment variable to
an existing CSV path before running the notebook.

The CSV must contain these columns: `cement`, `slag`, `flyash`, `water`,
`superplasticizer`, `coarseaggregate`, `fineaggregate`, `age`, and `csMPa`.
The notebook validates the columns, numeric values, missing values, and duplicate
rows before training.

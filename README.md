# Concrete Strength Prediction

This repository contains one Jupyter notebook for exploratory plots and regression experiments on the Yeh concrete compressive-strength dataset. The checked-in notebook analyzes the data and compares a train/test split for linear regression, random forest, and (optionally) XGBoost.

## Dataset

The dataset is **not included** in this repository. The notebook expects a CSV named `Concrete_Data_Yeh.csv` in either the repository root or a `data/` subdirectory, relative to the Jupyter working directory. Obtain the dataset from a source you are authorized to use; this repository does not provide or verify a download source. The CSV must contain these columns:

`cement`, `slag`, `flyash`, `water`, `superplasticizer`, `coarseaggregate`, `fineaggregate`, `age`, and `csMPa`.

If the file is absent or required columns are missing, the notebook reports a clear error instead of silently using a machine-specific path.

## Setup and use

Python 3.8 or newer is recommended. From the repository root, install the notebook's core dependencies and start Jupyter:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook "Concrete Strength Prediction.ipynb"
```

Run cells from top to bottom after placing the CSV in one of the supported locations. XGBoost is an optional final experiment; install it separately with `python -m pip install xgboost` if desired. Without it, the linear-regression and random-forest sections remain available.

The notebook uses a fixed `random_state` for its train/test split and random forest. Its reported metrics are dependent on the supplied dataset and installed library versions. No performance result is asserted here: the data file is not present in this repository, and this change does not validate model performance.

## Scope and limitations

The notebook includes exploratory summaries/plots plus regression; the earlier README listed classification and clustering algorithms that are not implemented in the checked-in notebook. No dataset, trained model, or dependency lockfile is provided. The notebook's saved cell outputs have been cleared so old results are not mistaken for a current run.

## Project structure

- `Concrete Strength Prediction.ipynb` — analysis and modeling notebook
- `README.md` — setup, input data requirements, and limitations

## Author

Abdul Rahman Ahmed

# California Housing Model Training

This repository contains a simple Python project for a university assignment to train and evaluate regression models on the California housing dataset.

## Project purpose
- Train basic regression models (Linear Regression and Decision Tree) to predict `median_house_value`.
- Demonstrate basic data exploration, missing-value handling, categorical encoding, and model evaluation.

## Files
- `main.py` - Main script that loads the dataset, performs exploratory analysis, fills missing numeric values with medians, encodes categorical features, trains two models, and prints evaluation metrics.
- `housing_data.csv` - Input dataset (California housing dataset). Keep this file in the project root.

## Dependencies
Create and activate a virtual environment (optional but recommended). The project requires:
- Python 3.8+
- pandas
- matplotlib
- scikit-learn

You can install dependencies with pip. If you use the included virtual environment, replace `python` with the path to the project's Python executable.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install packages manually:

```powershell
pip install pandas matplotlib scikit-learn
```

## How to run
Run the main script from PowerShell. If you use the project virtual environment created above, run:

```powershell
& ".\.venv\Scripts\python.exe" "$(Resolve-Path .\main.py)"
```

Or without a virtual environment:

```powershell
python .\main.py
```

The script will print exploratory outputs, fill missing numeric values (median imputation), encode `ocean_proximity` with one-hot columns, train Linear Regression and Decision Tree regressors, and print MSE and R² for both.

## Notes and next steps
- The current missing-value strategy is simple median imputation. For reproducible pipelines, consider scikit-learn's `SimpleImputer` and `Pipeline`.
- Categorical encoding is done with `pandas.get_dummies`; you may prefer `OneHotEncoder` inside a pipeline for production code.
- Add model persistence (`joblib.dump`) if you need to save models.

## Academic use
This project was developed as part of a university assignment. If you submit the code, include proper references for the dataset and any external resources used.

---

If you want, I can also add a `requirements.txt`, a small `Makefile` or a proper scikit-learn pipeline and a saved model example. Which would you like next?
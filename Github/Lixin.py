"""
Random Forest modeling with hyperparameter optimization and SHAP interpretation
Author: Lixin
Python dependencies:
    xarray, pandas, numpy, scikit-learn, optuna, shap, scipy
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import optuna
import shap
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Utility functions
# -----------------------------
def correlation_coefficient(y_true, y_pred):
    """Pearson correlation coefficient as evaluation metric."""
    return pearsonr(y_true, y_pred)[0]


def rf_objective(trial, X, y):
    """Objective function for Optuna hyperparameter tuning (Random Forest)."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.7]),
        "random_state": 42,
        "n_jobs": -1,
    }

    model = RandomForestRegressor(**params)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        scores.append(correlation_coefficient(y[val_idx], preds))

    return np.mean(scores)


# -----------------------------
# Main modeling pipeline
# -----------------------------
def train_rf_with_shap(X, y, name, out_dir, n_trials=10):
    """
    Train Random Forest model with Optuna tuning and compute SHAP values.
    """

    X = X.values
    y = y.values

    # Optional scaling (RF is scale-invariant, but kept for consistency)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # -------- Hyperparameter optimization --------
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: rf_objective(trial, X, y),
        n_trials=n_trials,
        n_jobs=1
    )

    best_params = study.best_params
    print(f"Best parameters ({name}):", best_params)
    print(f"Best CV correlation ({name}):", study.best_value)

    # -------- Final model training --------
    final_model = RandomForestRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X, y)

    # Out-of-sample predictions (5-fold)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = np.zeros_like(y)
    for train_idx, val_idx in kf.split(X):
        model_cv = RandomForestRegressor(
            **best_params,
            random_state=42,
            n_jobs=-1
        )
        model_cv.fit(X[train_idx], y[train_idx])
        y_pred[val_idx] = model_cv.predict(X[val_idx])

    pd.DataFrame(y_pred, columns=["y_pred"]).to_csv(
        f"{out_dir}/y_pred_{name}.csv", index=False
    )

    # -------- SHAP interpretation --------
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer(X)

    pd.DataFrame(
        shap_values.values,
        columns=[f"feature_{i}" for i in range(X.shape[1])]
    ).to_csv(f"{out_dir}/shap_values_{name}.csv", index=False)

    return final_model, shap_values


if __name__ == "__main__":

    df = pd.read_excel(
        r"G:\01 Dorctor\20251108 LIXIN\20260512 Lixin\SHAP_data.xlsx",
        header=0
    )

    OUT_DIR = r"G:\01 Dorctor\20251108 LIXIN\20260512 Lixin\DATA"

    train_rf_with_shap(
        X=df.loc[:, "pH":"Alt"],
        y=df["TF"],
        name="TF",
        out_dir=OUT_DIR,
        n_trials=10
    )
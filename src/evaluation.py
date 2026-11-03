import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)
from sklearn.base import clone

def get_predictions(
    model,
    X: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """Return predictions and probabilities."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return y_pred, y_proba

def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    name: str
) -> dict:
    """Calculate main metrics."""
    y_pred, y_proba = get_predictions(model, X)

    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y, y_pred), 4),
        "F1-Score": round(f1_score(y, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y, y_proba), 4)
    }

def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    name: str,
    cv: int = 5,
    **fit_kwargs
) -> dict:
    """Run stratified cross-validation and return mean metrics."""
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    fold_scores = []

    for train_idx, val_idx in cv_strategy.split(X,y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model_clone = clone(model)
        model_clone.fit(X_tr, y_tr, **fit_kwargs)

        y_pred, y_proba = get_predictions(model_clone, X_val)

        fold_scores.append({
            "accuracy": accuracy_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred),
            "roc_auc": roc_auc_score(y_val, y_proba)
        })

    return {
        "Model": name,
        "Accuracy": round(np.mean([s["accuracy"] for s in fold_scores]), 4),
        "F1-Score": round(np.mean([s["f1"] for s in fold_scores]), 4),
        "ROC-AUC": round(np.mean([s["roc_auc"] for s in fold_scores]), 4)
    }

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report
)

def get_predictions(
    model: BaseEstimator,
    X: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """Return predictions and probabilities."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return y_pred, y_proba

def evaluate_model(
    model: BaseEstimator,
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

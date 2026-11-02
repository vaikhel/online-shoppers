import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import get_preprocessor

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numerical_features: list,
    categorical_features: list
) -> Pipeline:
    """Train Logistic Regression with preprocessing."""
    pipeline = Pipeline(steps=[
        ("preprocessor", get_preprocessor(numerical_features, categorical_features)),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numerical_features: list,
    categorical_features: list
) -> Pipeline:
    """Train Decision Tree with preprocessing."""
    pipeline = Pipeline(steps=[
        ("preprocessor", get_preprocessor(numerical_features, categorical_features)),
        ("model", DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

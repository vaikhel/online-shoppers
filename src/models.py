import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from catboost import CatBoostClassifier

from src.preprocessing import get_preprocessor

def get_logistic_regression(
    numerical_features: list,
    categorical_features: list
) -> Pipeline:
    """Train Logistic Regression with preprocessing."""
    return Pipeline(steps=[
        ("preprocessor", get_preprocessor(numerical_features, categorical_features)),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

def get_decision_tree(
    numerical_features: list,
    categorical_features: list
) -> Pipeline:
    """Train Decision Tree with preprocessing."""
    return Pipeline(steps=[
        ("preprocessor", get_preprocessor(numerical_features, categorical_features)),
        ("model", DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=42
        ))
    ])

class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MLPWrapper:
    def __init__(
        self,
        numerical_features: list,
        categorical_features: list,
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 0.001
    ):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.preprocessor = None
        self.model = None

    def get_params(self, deep=True) -> dict:
        return {
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MLPWrapper":
        torch.manual_seed(42)

        self.preprocessor = get_preprocessor(self.numerical_features, self.categorical_features)
        X_processed = self.preprocessor.fit_transform(X)

        X_tensor = torch.tensor(
            X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed,
            dtype = torch.float32
        )
        y_tensor = torch.tensor(y.values, dtype=torch.long)

        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True
        )

        self.model = MLP(input_dim=X_tensor.shape[1])
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for _ in range(self.epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_processed = self.preprocessor.transform(X)
        X_tensor = torch.tensor(
            X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed,
            dtype=torch.float32
        )
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            proba = torch.softmax(logits, dim=1).numpy()
        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

def get_mlp(
        numerical_features: list,
        categorical_features: list,
        **kwargs
) -> MLPWrapper:
    return MLPWrapper(numerical_features, categorical_features, **kwargs)

def get_catboost() -> CatBoostClassifier:
    """Train CatBoost (handles categoricals natively)."""
    return CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        random_state=42,
        verbose=0,
        auto_class_weights="Balanced"
    )

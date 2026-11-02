import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer

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
    def __init__(self, model: nn.Module, preprocessor: ColumnTransformer):
        self.model = model
        self.preprocessor = preprocessor
        self.model.eval()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_processed = self.preprocessor.transform(X)
        X_tensor = torch.tensor(
            X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed,
            dtype=torch.float32
        )
        with torch.no_grad():
            logits = self.model(X_tensor)
            proba = torch.softmax(logits, dim=1).numpy()
        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

def train_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numerical_features: list,
    categorical_features: list,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 0.001
) -> MLPWrapper:
    """Train MLP and return a wrapper with predict/predict_proba."""
    torch.manual_seed(42)

    preprocessor = get_preprocessor(numerical_features, categorical_features)
    X_processed = preprocessor.fit_transform(X_train)
    X_tensor = torch.tensor(
        X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed,
        dtype=torch.float32
    )
    y_tensor = torch.tensor(y_train.values, dtype=torch.long)

    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=True
    )

    model = MLP(input_dim=X_tensor.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

    return MLPWrapper(model, preprocessor)

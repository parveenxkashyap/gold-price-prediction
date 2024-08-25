from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class ModelResult:
    model: LinearRegression
    predictions: pd.Series


def train_linear_regression(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    x_col: str = "LaggedReturn",
    y_col: str = "Return",
) -> ModelResult:
    if x_col not in train_df.columns or y_col not in train_df.columns:
        raise KeyError(f"Train df must contain '{x_col}' and '{y_col}'")

    if x_col not in test_df.columns or y_col not in test_df.columns:
        raise KeyError(f"Test df must contain '{x_col}' and '{y_col}'")

    x_train = train_df[[x_col]]
    y_train = train_df[y_col]

    x_test = test_df[[x_col]]

    model = LinearRegression()
    model.fit(x_train, y_train)

    preds = pd.Series(model.predict(x_test), index=test_df.index, name="Predicted")
    return ModelResult(model=model, predictions=preds)

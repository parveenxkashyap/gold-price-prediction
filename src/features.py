from __future__ import annotations

import pandas as pd


def add_returns_and_lag(
    df: pd.DataFrame,
    price_col: str = "USD PM",
    return_col: str = "Return",
    lag_col: str = "LaggedReturn",
) -> pd.DataFrame:
    """
    Create:
    - Return: percent change * 100 of the given price column
    - LaggedReturn: Return shifted by 1
    """
    if price_col not in df.columns:
        raise KeyError(
            f"Missing required column '{price_col}'. Available columns: {list(df.columns)}"
        )

    out = df.copy()
    out[return_col] = out[price_col].pct_change() * 100.0
    out[lag_col] = out[return_col].shift(1)
    out = out.dropna(subset=[return_col, lag_col])
    return out


def split_train_test_by_year(
    df: pd.DataFrame,
    train_start: str = "2001",
    train_end: str = "2018",
    test_year: str = "2019",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Slice a DateTimeIndex dataframe using pandas time slicing:
    - train: [train_start .. train_end]
    - test: [test_year]
    """
    train = df.loc[train_start:train_end]
    test = df.loc[test_year:test_year]
    return train, test

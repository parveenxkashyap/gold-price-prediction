from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_gold_prices(csv_path: str | Path, date_col: str = "Date") -> pd.DataFrame:
    """
    Load historical gold prices from a CSV.

    Expected to match the notebook-style layout:
    - Date column used as index
    - price columns such as 'USD PM'
    """
    csv_path = Path(csv_path)

    df = pd.read_csv(csv_path, parse_dates=[date_col])
    df = df.set_index(date_col).sort_index()
    return df

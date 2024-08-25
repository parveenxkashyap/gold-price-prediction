from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_actual_vs_predicted(
    df: pd.DataFrame,
    actual_col: str = "Actual",
    predicted_col: str = "Predicted",
    title: str = "Gold returns: actual vs predicted (2019)",
    outpath: str | Path = "outputs/predictions_2019.png",
) -> Path:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    ax = df[[actual_col, predicted_col]].plot(figsize=(10, 4), title=title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Return (%)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

    return outpath

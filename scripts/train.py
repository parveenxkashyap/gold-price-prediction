from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gold_price_prediction.data import load_gold_prices
from gold_price_prediction.features import add_returns_and_lag, split_train_test_by_year
from gold_price_prediction.modeling import train_linear_regression


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train LinearRegression on lagged returns and predict 2019 returns.")
    p.add_argument("--csv", type=str, required=True, help="Path to goldprice.csv")
    p.add_argument("--outdir", type=str, default="outputs", help="Directory to save outputs")
    p.add_argument("--price-col", type=str, default="USD PM", help="Price column to use (default: 'USD PM')")
    p.add_argument("--train-start", type=str, default="2001", help="Train start (time slice)")
    p.add_argument("--train-end", type=str, default="2018", help="Train end (time slice)")
    p.add_argument("--test-year", type=str, default="2019", help="Test year (time slice)")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_raw = load_gold_prices(args.csv)
    df_feat = add_returns_and_lag(df_raw, price_col=args.price_col)

    train_df, test_df = split_train_test_by_year(
        df_feat,
        train_start=args.train_start,
        train_end=args.train_end,
        test_year=args.test_year,
    )

    result = train_linear_regression(train_df, test_df)

    out = pd.DataFrame(
        {
            "Actual": test_df["Return"],
            "Predicted": result.predictions,
        }
    )
    out_path = outdir / "predictions_2019.csv"
    out.to_csv(out_path, index=True)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

# Gold Price Prediction (Linear Regression)

This project uses historical gold prices to predict next-day returns using a simple linear regression model.

It follows the same idea as the notebook:
- Load historical gold prices from CSV
- Compute daily % returns and a 1-day lagged return feature
- Train on 2001–2018 and test on 2019
- Plot actual vs. predicted returns for 2019

## Project structure

├── data/
│ └── goldprice.csv # you provide this file
├── scripts/
│ └── train.py # CLI to train + evaluate
├── src/
│ └── gold_price_prediction/
│ ├── init.py
│ ├── data.py
│ ├── features.py
│ ├── modeling.py
│ └── plotting.py
├── requirements.txt
└── notebook.ipynb # original notebook
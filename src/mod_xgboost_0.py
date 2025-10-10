# Importing packages
import country_converter as coco
import matplotlib.pyplot as plt
import xgboost as xgb
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import csv

# Importing partial packages
from sklearn.model_selection import train_test_split, GridSearchCV
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Defining Custom Functions
def create_lag(neighbor, dataframe):
    # Create lag features (e.g., previous 5 days' returns)
    for i in range(1, 25):
        dataframe[f"return_lag_{i}"] = dataframe[neighbor].pct_change().shift(i)

    # Create moving average features
    dataframe["MA_30"] = dataframe[neighbor].rolling(window=30).mean().shift(1)
    dataframe["MA_60"] = dataframe[neighbor].rolling(window=60).mean().shift(1)
    dataframe["MA_90"] = dataframe[neighbor].rolling(window=90).mean().shift(1)

    # Drop initial rows with NaNs created by rolling windows
    dataframe.dropna(inplace=True)
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan).fillna(0)

    return dataframe

# Run this for the neighbor alone: control
# Run this for the anchor + neighbor: treatment
def perform_xgb_for_anchor_neighbor(neighbor, dataframe):
    # Starting with CNY/CDF as an example
    dataframe["target"] = dataframe[neighbor].shift(-1)

    # Define features (X) and target (y)
    features = [
        col
        for col in df_0.columns
        if col == neighbor or col.startswith("MA_") or col.startswith("return_lag_")
    ]
    X = dataframe[features]
    y = dataframe["target"]

    # Time-based train-test split
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    xgb_0 = xgb.XGBRegressor()
    parameters = {
        "nthread": [4],
        "objective": ["reg:squarederror"],
        "learning_rate": [0.08, 0.15, 0.3],
        "max_depth": [2, 4, 8],
        "min_child_weight": [4],
        "subsample": [0.2, 0.5, 0.8],
        "colsample_bytree": [0.7],
        "n_estimators": [2, 5, 10],
    }

    xgb_grid = GridSearchCV(xgb_0, parameters, cv=5, n_jobs=5, verbose=True)
    xgb_grid.fit(X_train, y_train)

    return xgb_grid.best_score_, xgb_grid.best_params_


if __name__ == "__main__":
    # Instantiating neighbor groups
    ng = {'EUR': ['ALL', 'CZK', 'TND', 'RON', 'RSD', 'DZD', 'HUF', 'SEK', 'ISK', 'PLN']}
    ls_0 = [i for i in ng.items() if i[0] == "EUR"][0]
    ls_1 = [ls_0[0] for i in range(len([i for i in ng.items() if i[0] == "EUR"][0][1]))]
    xgb_df = pd.DataFrame({"anchor": ls_1, "neighbor": ls_0[1]})

    df_0 = pd.read_parquet("../data/input/fx_log_return.parquet")

    for i, j in tqdm(zip(xgb_df["anchor"], xgb_df["neighbor"])):
        df_1 = df_0[[i, j]]
        df_1.dropna(inplace=True)
        df_1 = create_lag(j, df_1).drop(columns=[i])
        score, params = perform_xgb_for_anchor_neighbor(j, df_1)

        print(f"Anchor: {i}, Neighbor: {j}, Score: {score}, Params: {params}")
        print()
        print()
        print()
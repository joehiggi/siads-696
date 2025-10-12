# Importing packages
import country_converter as coco
import matplotlib.pyplot as plt
import xgboost as xgb
import yfinance as yf
import pandas as pd
import numpy as np
import csv

# Importing partial packages
from sklearn.model_selection import train_test_split, GridSearchCV
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
from utility import create_lag, load_data
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from tqdm import tqdm


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
	# Importing data
    neighbor_groups, df_anchor_neighor, df_log_return_pca = load_data()

    for i, j in tqdm(zip(df_anchor_neighor["anchor"], df_anchor_neighor["neighbor"]), total=len(df_anchor_neighor)):
		
        # Preparing data for train test split
        df_1 = df_log_return_pca[[i, j]]
        df_1.dropna(inplace=True)
        df_1 = create_lag(j, df_1).drop(columns=[i])
        
        score, params = perform_xgb_for_anchor_neighbor(j, df_1)        
        print(f"Anchor: {i}, Neighbor: {j}, Score: {score}, Params: {params}")
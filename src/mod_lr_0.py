# Importing packages
import pandas as pd
import numpy as np
import warnings

# Importing partial packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# Ignoring warnings
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

if __name__ == "__main__":
	# Instantiating neighbor groups
	ng = {"EUR": ["ALL", "CZK", "TND", "RON", "RSD", "DZD", "HUF", "SEK", "ISK", "PLN"]}
	ls_0 = [i for i in ng.items() if i[0] == "EUR"][0]
	ls_1 = [ls_0[0] for i in range(len([i for i in ng.items() if i[0] == "EUR"][0][1]))]
	df_2 = pd.DataFrame({"anchor": ls_1, "neighbor": ls_0[1]})
	df_0 = pd.read_parquet("../data/input/fx_log_return.parquet")
    
	# Running LR for each anchor neighbor pair
	for i, j in zip(df_2["anchor"], df_2["neighbor"]):

		# Preparing data for train test split
		df_1 = df_0[[i, j]]
		df_1.dropna(inplace=True)
		df_1 = create_lag(j, df_1).drop(columns=[i])
		
		# Time-based train-test split
		df_1["target"] = df_1[j].shift(-1)
		df_1.dropna(inplace=True)

		# Define features (X) and target (y)
		features = [col for col in df_1.columns if col == j or col.startswith("MA_") or col.startswith("return_lag_")]
		X = df_1[features]
		y = df_1["target"]
        
		# Splitting data into train and test sets
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
		clf_lr = LinearRegression().fit(X, y)
		y_preds = clf_lr.predict(X_test)
		# print(y_test, y_preds)
		print(i, j, clf_lr.score(X_test, y_test))
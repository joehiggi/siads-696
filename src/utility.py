# Importing packages
import pandas as pd
import numpy as np
import warnings

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

def load_data():
    # Instantiating neighbor groups
    neighbor_groups = {"EUR": ["ALL", "CZK", "TND", "RON", "RSD", "DZD", "HUF", "SEK", "ISK", "PLN"]}
    
    # Instantiating anchor neighbor df
    ls_0 = [i for i in neighbor_groups.items() if i[0] == "EUR"][0]
    ls_1 = [ls_0[0] for i in range(len([i for i in neighbor_groups.items() if i[0] == "EUR"][0][1]))]
    df_anchor_neighor = pd.DataFrame({"anchor": ls_1, "neighbor": ls_0[1]})

    # Loading stored pca df
    df_log_return_pca = pd.read_parquet("../data/input/fx_log_return.parquet")
    
    return neighbor_groups, df_anchor_neighor, df_log_return_pca


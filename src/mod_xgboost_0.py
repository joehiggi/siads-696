# Importing packages
import xgboost as xgb
import pandas as pd
import numpy as np

# Importing partial packages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
from utility import create_lag, load_data
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from tqdm import tqdm


def perform_xgb_for_anchor_neighbor(neighbor, dataframe, X, y):

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
    y_preds = xgb_grid.predict(X_test)

    return xgb_grid.best_score_, xgb_grid.best_params_, y_preds

if __name__ == "__main__":
	# Importing data
    neighbor_groups, df_anchor_neighor, df_log_return_pca = load_data()

    # Compiling scores
    final = []

    # Running LR for each anchor neighbor pair
    for option in ["Training_with_Anchor", "Training_without_Anchor"]:

        for i, j in tqdm(zip(df_anchor_neighor["anchor"], df_anchor_neighor["neighbor"]), total=len(df_anchor_neighor)):
		
            # Initializing dictionary to hold scores
            scores = {}

            # Preparing data for train test split
            df_1 = df_log_return_pca[[i, j]]
            df_1 = df_1.drop(columns=[i])
            df_1 = create_lag(j, df_1)
            
            # Time-based train-test split
            df_1["target"] = df_1[j].shift(-1)
            df_1.dropna(inplace=True)

            # Define features (X) and target (y)
            if option == "Training_without_Anchor":
                features = [col for col in df_1.columns]
                i = None
            else:
                features = [col for col in df_1.columns if col == j or col.startswith("MA_") or col.startswith("return_lag_")]
                
            X = df_1[features]
            y = df_1["target"]

            # Preparing data for train test split
            df_1 = df_log_return_pca[[i, j]]
            df_1.dropna(inplace=True)
            df_1 = create_lag(j, df_1).drop(columns=[i])
            score, params, y_preds = perform_xgb_for_anchor_neighbor(j, df_1, X, y)

            # Assigning scores
            scores['anchor'] = i
            scores['neighbor'] = j
            scores['option'] = option
            # scores['rmse'] = root_mean_squared_error(y_test, y_preds)
            scores['rmse'] = np.mean(np.sqrt(np.abs(cross_val_score(xgb.XGBRegressor(**params), X, y, scoring = 'neg_mean_squared_error', cv = 5, n_jobs = -1))))
            scores['model'] = 'XGBoost'   
            scores['params'] = None        
            
            # print(scores)
            final.append(scores)
        
    df_final = pd.DataFrame(final)
    df_final_summary_mean = df_final.groupby('option')['rmse'].mean()
    df_final_summary_std = df_final.groupby('option')['rmse'].std()
    print(df_final)
    print(df_final_summary_mean)
    print(df_final_summary_std)
    
    print(rf"Difference without anchor: {df_final_summary_mean[0] - df_final_summary_mean[1]}") 

    # print(f"Anchor: {i}, Neighbor: {j}, Score: {score}, Params: {params}", y_preds)
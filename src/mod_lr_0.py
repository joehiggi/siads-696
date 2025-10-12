# Importing packages
import pandas as pd
import numpy as np

# Importing partial packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression
from utility import create_lag, load_data
from tqdm import tqdm

if __name__ == "__main__":
	
	# Importing data
    neighbor_groups, df_anchor_neighor, df_log_return_pca = load_data()

	# Compiling scores
    final = []

	# Running LR for each anchor neighbor pair
    for option in ["Training_with_Anchor", "Training_without_Anchor"]:
        
        for i, j in tqdm(zip(df_anchor_neighor["anchor"], df_anchor_neighor["neighbor"]), total=len(df_anchor_neighor)):    
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
            
			# Splitting data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf_lr = LinearRegression().fit(X, y)
            y_preds = clf_lr.predict(X_test)
            
            # Assigning scores
            scores['anchor'] = i
            scores['neighbor'] = j
            scores['option'] = option
            scores['rmse'] = root_mean_squared_error(y_test, y_preds)
            scores['model'] = 'Linear Regression'            
            
            # print(scores)
            final.append(scores)
        
        df_final = pd.DataFrame(final)
        print(df_final)
        print(df_final.groupby('option')['rmse'].std())
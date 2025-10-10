# Importing packages
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import numpy as np
import warnings
import torch

# Importing partial packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
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

# Defining the LSTM model
class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)

        return out


if __name__ == "__main__":
    # Instantiating neighbor groups
    ng = {"EUR": ["ALL", "CZK", "TND", "RON", "RSD", "DZD", "HUF", "SEK", "ISK", "PLN"]}
    ls_0 = [i for i in ng.items() if i[0] == "EUR"][0]
    ls_1 = [ls_0[0] for i in range(len([i for i in ng.items() if i[0] == "EUR"][0][1]))]
    df_2 = pd.DataFrame({"anchor": ls_1, "neighbor": ls_0[1]})
    df_0 = pd.read_parquet("../data/input/fx_log_return.parquet")

    # Running LSTM for each anchor-neighbor pair
    for i, j in tqdm(zip(df_2["anchor"], df_2["neighbor"])):

        # Preparing data for train test split
        df_1 = df_0[[i, j]]
        df_1.dropna(inplace=True)
        df_1 = create_lag(j, df_1).drop(columns=[i])
        
        # Time-based train-test split
        df_1["target"] = df_1[j].shift(-1)
        
        # Define features (X) and target (y)
        features = [col for col in df_1.columns if col == j or col.startswith("MA_") or col.startswith("return_lag_")]
        X = df_1[features]
        y = df_1["target"]

        # Splitting training data
        train_size = int(len(X) * 0.8)
        train_X = Variable(torch.Tensor(np.array(X[0:train_size]))).unsqueeze(2) # https://stackoverflow.com/questions/57237352/what-does-unsqueeze-do-in-pytorch
        train_y = Variable(torch.Tensor(np.array(y[0:train_size])))
        
        # Splitting testing data
        test_size = len(y) - train_size
        test_X = Variable(torch.Tensor(np.array(X[train_size:len(X)]))).unsqueeze(2) # https://stackoverflow.com/questions/57237352/what-does-unsqueeze-do-in-pytorch
        test_y = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

        num_epochs = 2000
        learning_rate = 0.01
        input_size = 1
        hidden_size = 2
        num_layers = 1
        num_classes = 1
        seq_length = len(train_X)

        lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

        criterion = torch.nn.MSELoss()  # mean-squared error for regression
        optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

        # Train the model
        for epoch in tqdm(range(num_epochs)):
            outputs = lstm(train_X)
            optimizer.zero_grad()

            # obtain the loss function
            loss = criterion(
                outputs, train_y
            )
            loss.backward()

            optimizer.step()
            if epoch % 100 == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

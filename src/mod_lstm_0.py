# Importing packages
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import numpy as np
import torch

# Importing partial packages
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from utility import create_lag, load_data
from torch.autograd import Variable
from tqdm import tqdm

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

	# Importing data
    neighbor_groups, df_anchor_neighor, df_log_return_pca = load_data()

    # Compiling scores
    final = []

    # Running LR for each anchor neighbor pair
    for option in ["Training_with_Anchor", "Training_without_Anchor"]:

        # Running LSTM for each anchor-neighbor pair
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
                loss = torch.sqrt(criterion(
                    outputs, train_y
                ))
                loss.backward()

                optimizer.step()
                if epoch % 100 == 0:
                    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

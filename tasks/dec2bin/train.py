"""
PREREQUISITE: Run tasks/dec2bin/create_dataset.py to generate dec2bin_train.csv and dec2bin_test.csv

This train loop uses an RNN, where the input integer is fed in once, then 0s are fed in, once for each bit.
Crude approach, as we don't have a one-to-many implementation of RNN.

"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from neural_verification import RNNConfig, RNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert binary string to tensor
def binary_str_to_tensor(s):
    return torch.tensor([int(b) for b in str(s)], dtype=torch.float32).to(device)

class Dec2Bin(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Dec2Bin, self).__init__()
        self.fcFirst = nn.Linear(input_dim, hidden_dim)
        self.hiddens = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(4)])
        self.fcLast = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fcFirst(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = F.relu(x)

        x = self.fcLast(x)
        return self.sigmoid(x)

    
def print_grads(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Gradient of {name}: {param.grad.data.norm(2).item()}")

if __name__ == "__main__":
    df_train = pd.read_csv("dec2bin_train.csv", dtype={'integer': int, 'binary': str})
    df_test = pd.read_csv("dec2bin_test.csv", dtype={'integer': int, 'binary': str})

    print(df_train.head())  # Check a few rows to ensure the data looks as expected

    # Convert to tensors
    train_data = [(torch.tensor([row['integer']]) / (2**16), binary_str_to_tensor(row['binary'])) for _, row in df_train.iterrows()]
    test_data = [(torch.tensor([row['integer']]) / (2**16), binary_str_to_tensor(row['binary'])) for _, row in df_test.iterrows()]

    print(train_data[0])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Initialize model
    input_dim = 1  # Integer as input
    hidden_dim = 64
    output_dim = 16  # Length of binary string
    model = Dec2Bin(input_dim, hidden_dim, output_dim).to(device)

    # BCE loss
    criterion = nn.BCELoss()

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.03)

    # Train loop

    steps = 200
    pbar = tqdm(total=steps)
    train_losses = []
    test_losses = []

    train_accuracies = []
    test_accuracies = []

    for step in range(steps):
        model.train()
        correct, total = 0, 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device).float()  # batch_size x sequence_length x input_dim
            y = y.to(device)
            optimizer.zero_grad()
            output = model(x).squeeze(-1)  # Remove the last dimension of size 1

            loss = criterion(output, y) 
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            y_pred_binary = (output > 0.5).float()
            correct += (y_pred_binary == y).all(dim=1).float().sum()
            total += y.shape[0]

            debug = False
            if debug and batch_idx % 100 == 0:  # Adjust frequency as needed
                print("Output shape:", output.shape)
                print("Y shape:", y.shape)
                print("Sample Input:", x[0])
                print("Sample Output:", y_pred_binary[0])
                print("Ground Truth:", y[0])

                print_grads(model)

        train_accuracies.append(correct / total)
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device).float()  # batch_size x sequence_length x input_dim
                y = y.to(device)
                output = model(x).squeeze(-1)
                loss = criterion(output, y)
                test_losses.append(loss.item())

                y_pred_binary = (output > 0.5).float()
                correct += (y_pred_binary == y).sum(dim=1).eq(y.size(1)).sum().item()
                total += y.shape[0]

        test_accuracies.append(correct / total)


        pbar.set_description(f"tr l {train_losses[-1]:.2e} te l {test_losses[-1]:.2e} tr a {train_accuracies[-1]:.2} te a {test_accuracies[-1]:.2}")
        pbar.update(1)

    pbar.close()

    metrics = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_accuracies": train_accuracies,
        "test_accuracies": test_accuracies
    }


    # Save metrics and model
    torch.save(model.state_dict(), "dec2bin_model.pt")
    torch.save(metrics, "dec2bin_metrics.pt")


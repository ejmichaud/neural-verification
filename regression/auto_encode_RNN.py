import numpy as np
import torch
import os
import pandas as pd
from neural_verification import *
import matplotlib.pyplot as plt
import itertools
import copy
from sklearn.linear_model import LinearRegression
import argparse

from integer_autoencoder import *

# Create arguments for the script
parser = argparse.ArgumentParser(description='Specifications To Save Hidden Layer')
parser.add_argument('--task', type=str, default='rnn_identity_numerical', help='task name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--device', type=str, default='cpu', help='device')

args = parser.parse_args()
task = args.task
seed = args.seed
device = args.device

np.random.seed(seed)
torch.manual_seed(seed)
device=torch.device(device)

data = torch.load(f"../tasks/{task}/data.pt")
config = torch.load(f"../tasks/{task}/model_config.pt", map_location=torch.device('cpu'))
model = GeneralRNN(config, device=torch.device('cpu'))
model.load_state_dict(torch.load(f"../tasks/{task}/model_perfect.pt", map_location=torch.device('cpu')))

# Function to get hidden states
def get_hidden(model, x):
    # x shape: (batch_size, sequence_length, input_dim)
    batch_size = x.size(0)
    seq_length = x.size(1)
    hidden = torch.zeros(batch_size, model.hidden_dim).to(device)
    outs = []
    for i in range(seq_length):
        out, hidden = model.forward(x[:,i,:], hidden)
        if i == seq_length - 2:
            hidden_second_last = hidden.clone()
        outs.append(out)
    return torch.stack(outs).permute(1,0,2), hidden.cpu().detach().numpy(), hidden_second_last.cpu().detach().numpy()

_, hidden, hidden2 = get_hidden(model, data[0].unsqueeze(dim=2))
inputs_last = data[0][:,[-1]].cpu().detach().numpy()

# Function to update task status in a CSV file
def update_task_status(task_name, Q1, Q2):
    file_path = 'task_status.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if task_name in df["Task"].tolist():
            if 'Q1' not in df.columns:
                df['Q1'] = np.nan
            if 'Q2' not in df.columns:
                df['Q2'] = np.nan
            df.loc[df['Task'] == task_name, 'Q1'] = Q1
            df.loc[df['Task'] == task_name, 'Q2'] = Q2
            df.to_csv(file_path, index=False)
        else:
            print(f"Task {task_name} not found in {file_path}")
    else:
        print(f"File {file_path} not found")

# Rest of your existing code...
# LinRNNautoencode
A_lin, b_lin, Z_lin, Z_lin2 = LinRNNautoencode(hidden, hidden2, inputs_last)
Q1_lin, Q2_lin = integer_autoencoder_quality(hidden, A_lin, b_lin, Z_lin)

# GCDautoencode with hyperparameter tuning
fractions = 10 ** np.linspace(-3, 0, num=10)
thresholds = 10 ** np.linspace(-3, 0, num=10)

F, T = np.meshgrid(fractions, thresholds)
hyperparams = np.transpose(np.array([F.reshape(-1), T.reshape(-1)]))
min_dl = float('inf')
optimal_hp = None
gcd_success = False

for hp in hyperparams:
    try:
        A_gcd, b_gcd, Z_gcd, Z_gcd2 = GCDautoencode(hidden, hidden2, thres=hp[0], fraction=hp[1])
        dl, _ = integer_autoencoder_quality(hidden, A_gcd, b_gcd, Z_gcd)
        if dl < min_dl:
            min_dl = dl
            optimal_hp = hp
            gcd_success = True
    except Exception as e:
        last_error_message = str(e)


if not gcd_success:
    print(f"GCD Failed for {task}")
    print(last_error_message)
    
# Evaluate the optimal GCDautoencode
print(optimal_hp)
A_gcd, b_gcd, Z_gcd, Z_gcd2 = GCDautoencode(hidden, hidden2, thres=optimal_hp[0], fraction=optimal_hp[1])
Q1_gcd, Q2_gcd = integer_autoencoder_quality(hidden, A_gcd, b_gcd, Z_gcd)

# Choose the method with the best Q1 score and update the file
if Q1_gcd > Q1_lin:
    # Update file with LinRNNautoencode results
    A_best, b_best, Z_best, Z2_best = A_lin, b_lin, Z_lin, Z_lin2
    Q1_best, Q2_best = Q1_lin, Q2_lin
else:
    # Update file with GCDautoencode results
    A_best, b_best, Z_best, Z2_best = A_gcd, b_gcd, Z_gcd, Z_gcd2
    Q1_best, Q2_best = Q1_gcd, Q2_gcd

print(f"Updating Spreadsheet with {task, Q1_best, Q2_best}")
update_task_status(task, Q1_best, Q2_best)


def save_tensors(task, A, b, Z, Z2):
    os.makedirs(f"./tasks/{task}", exist_ok=True)
    torch.save(A, f"./tasks/{task}/A_best.pt")
    torch.save(b, f"./tasks/{task}/b_best.pt")
    torch.save(Z, f"./tasks/{task}/Z_best.pt")
    torch.save(Z2, f"./tasks/{task}/Z2_best.pt")

# Function to save tensors in a CSV file
def save_to_csv(task, A, b, Z, Z2):
    A_df = pd.DataFrame(A)
    b_df = pd.DataFrame(b)
    Z_df = pd.DataFrame(Z)
    Z2_df = pd.DataFrame(Z2)
    combined_df = pd.concat([A_df, b_df, Z_df, Z2_df], axis=1)
    combined_df.columns = [f"A_{i}" for i in range(A_df.shape[1])] + [f"b_{i}" for i in range(b_df.shape[1])] + [f"Z_{i}" for i in range(Z_df.shape[1])] + [f"Z2_{i}" for i in range(Z_df.shape[1])]
    combined_df.to_csv(f"./tasks/{task}/tensors.csv", index=False)

# Save the best tensors as .pt files and in a CSV file
save_tensors(task, A_best, b_best, Z_best, Z2_best)
save_to_csv(task, A_best, b_best, Z_best, Z2_best)
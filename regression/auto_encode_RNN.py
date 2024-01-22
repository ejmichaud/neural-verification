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
    # if x has shape (batch_size, sequence length), make it ((batch_size, sequence length, 1))
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

if len(data[0].shape) == 2:
    inputs = data[0].unsqueeze(dim=2)
else:
    inputs = data[0]
    
_, hidden, hidden2 = get_hidden(model, inputs)


if len(data[0].shape) == 2:
    inputs_last = data[0][:,[-1]].cpu().detach().numpy()
else:
    inputs_last = data[0][:,-1,:].cpu().detach().numpy()

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
# GCD autoencode is very expensive for high-dim space (skip when hidden dim >= 4)
if model.hidden_dim < 4:
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
else:
    Q1_gcd, Q2_gcd = 1e20, 1e20

# Choose the method with the best Q1 score and update the file
if Q1_gcd > Q1_lin:
    # Update file with LinRNNautoencode results
    A_best, b_best, Z_best, Z2_best = A_lin, b_lin, Z_lin, Z_lin2
    Q1_best, Q2_best = Q1_lin, Q2_lin
else:
    # Update file with GCDautoencode results
    A_best, b_best, Z_best, Z2_best = A_gcd, b_gcd, Z_gcd, Z_gcd2
    Q1_best, Q2_best = Q1_gcd, Q2_gcd

#print(f"Updating Spreadsheet with {task, Q1_best, Q2_best}")
#update_task_status(task, Q1_best, Q2_best)


def save_tensors(task, A, b, Z, Z2, hidden, hidden2):
    os.makedirs(f"./tasks/{task}", exist_ok=True)
    torch.save(A, f"./tasks/{task}/A_best.pt")
    torch.save(b, f"./tasks/{task}/b_best.pt")
    torch.save(Z, f"./tasks/{task}/Z_best.pt")
    torch.save(Z2, f"./tasks/{task}/Z2_best.pt")
    torch.save(hidden, f"./tasks/{task}/hidden.pt")
    torch.save(hidden2, f"./tasks/{task}/hidden2.pt")

# Function to save tensors in a CSV file
def save_to_csv(task, A, b, Z, Z2):
    A_df = pd.DataFrame(A)
    b_df = pd.DataFrame(b)
    Z_df = pd.DataFrame(Z)
    Z2_df = pd.DataFrame(Z2)
    combined_df = pd.concat([A_df, b_df, Z_df, Z2_df], axis=1)
    combined_df.columns = [f"A_{i}" for i in range(A_df.shape[1])] + [f"b_{i}" for i in range(b_df.shape[1])] + [f"Z_{i}" for i in range(Z_df.shape[1])] + [f"Z2_{i}" for i in range(Z_df.shape[1])]
    combined_df.to_csv(f"./tasks/{task}/tensors.csv", index=False)
    
def get_data(task_name):
    A, b, Z, Z2 = load_tensors(task_name)
    Z = np.round(Z); Z2 = np.round(Z2)
    data = torch.load(f"../tasks/{task_name}/data.pt")
    inputs = data[0].detach().numpy()
    outputs = data[1].detach().numpy()

    if len(data[0].shape) == 2:
        inputs_last = data[0][:,[-1]].detach().numpy()
    else:
        inputs_last = data[0][:,-1].detach().numpy()
    outputs_last = data[1][:,[-1]].detach().numpy()
    return A, b, Z, Z2, inputs, inputs_last, outputs, outputs_last

def load_tensors(task):
    # Construct the file paths
    a_path = f"./tasks/{task}/A_best.pt"
    b_path = f"./tasks/{task}/b_best.pt"
    z_path = f"./tasks/{task}/Z_best.pt"
    z2_path = f"./tasks/{task}/Z2_best.pt"

    # Load the tensors
    A = torch.load(a_path)
    b = torch.load(b_path)
    Z = torch.load(z_path)
    Z2 = torch.load(z2_path)

    return A, b, Z, Z2
    
def save_to_txt4sr(task_name):
    print('saving txt')
    A, b, Z, Z2, inputs, inputs_last, outputs, outputs_last = get_data(task_name)
    # (inputs_last, Z2) -> Z
    hidden_dim = Z.shape[1]
    # just save batch of it for efficiency
    # in codes try modifying .txt directly
    for i in range(hidden_dim):
        np.savetxt(f"./tasks/{task_name}/hidden{i}.txt", np.concatenate((Z2, inputs_last, Z[:,[i]]), axis=1), fmt="%d", delimiter=" ", newline=" \n")
    # Z -> outputs_last
    if len(outputs_last.shape) == 3:
        outputs_last = outputs_last[:,:,0]
    np.savetxt(f"./tasks/{task_name}/output.txt", np.concatenate((Z, outputs_last), axis=1), fmt="%d", delimiter=" ", newline=" \n")
    

# Save the best tensors as .pt files and in a CSV file
save_tensors(task, A_best, b_best, Z_best, Z2_best, hidden, hidden2)
save_to_csv(task, A_best, b_best, Z_best, Z2_best)
save_to_txt4sr(task)

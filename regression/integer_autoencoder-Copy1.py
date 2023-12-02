import numpy as np
import torch
import torch.nn as nn
import os
from dataclasses import dataclass
from neural_verification import *
import matplotlib.pyplot as plt
import itertools
import copy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import argparse
import pandas as pd



# Create arugments for the script
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
device = torch.device("cpu")



def get_hidden(model,x):
     #x shape: (batch_size, sequence_length, input_dim)
    batch_size = x.size(0)
    seq_length = x.size(1)
    hidden = torch.zeros(batch_size, model.hidden_dim).to(device)
    outs = []
    for i in range(seq_length):
        out, hidden = model.forward(x[:,i,:], hidden)
        if i == seq_length-2:
            hidden_second_last = hidden.clone()
        outs.append(out)
    return torch.stack(outs).permute(1,0,2), hidden.cpu().detach().numpy(), hidden_second_last.cpu().detach().numpy()


# Load data and model
data = torch.load(f"../tasks/{task}/data.pt", map_location=device)
config = torch.load(f"../tasks/{task}/model_config.pt", map_location=device)
model = GeneralRNN(config, device=device)
model.load_state_dict(torch.load(f"../tasks/{task}/model_perfect.pt",map_location=device))


# Extract data
input_data = data[0].cpu().detach().numpy()  # Assuming data[0] is a tensor
outputs, hidden_last, hidden_second_last = get_hidden(model,data[0].unsqueeze(dim=2))

# Convert tensors to numpy arrays
hidden_last = hidden_last.cpu().detach().numpy()
hidden_second_last = hidden_second_last.cpu().detach().numpy()
outs = outputs.cpu().detach().numpy()

output_dir = f"./tasks/{task}_hiddens"
os.makedirs(output_dir, exist_ok=True)
torch.save(hidden_last, os.path.join(output_dir, 'hidden_last.pt'))
torch.save(hidden_second_last, os.path.join(output_dir, 'hidden_second_last.pt'))
print(f"Successfully Saved Hidden Layers in {output_dir}")


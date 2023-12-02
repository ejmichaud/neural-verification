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
import pickle


# Create arugments for the script
parser = argparse.ArgumentParser(description='Specifications for Regression to Code')
parser.add_argument('--task', type=str, default='rnn_identity_numerical', help='task name')
parser.add_argument('--device', type=str, default='cpu', help='device')

args = parser.parse_args()
task = args.task
device = args.device

device = torch.device(device)

data = torch.load(f"../tasks/{task}/data.pt", map_location=device)
output_dir = f"./tasks/{task}_hiddens"

with open(os.path.join(output_dir, 'hidden_to_hidden.pkl'), 'rb') as file:
    reg1 = pickle.load(file)

with open(os.path.join(output_dir, 'hidden_to_output.pkl'), 'rb') as file:
    reg2 = pickle.load(file)
    
X = data[0].unsqueeze(dim=2)

batch_size = X.size(0)
seq_length = X.size(1)

#hack
hidden_dim = 1
hidden = torch.zeros(batch_size, hidden_dim)

outs = []

#+1 to go another iteration to match the dataset
for i in range(seq_length):
    hx = torch.cat((hidden, X[:,i,:]), dim=1)
    out, hidden = torch.tensor(reg2.predict(hidden)), torch.tensor(reg1.predict(hx))
    out=out.unsqueeze(1)
    outs.append(out)
outs, hidden = torch.stack(outs).permute(1,0,2), hidden

outs = outs.squeeze(2)
print(f"{((outs.int() == data[1]).float().mean().item()):.2f}")

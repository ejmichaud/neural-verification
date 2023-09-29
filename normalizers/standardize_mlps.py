import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import lr_scheduler
import numpy as np
from numpy import random
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn import metrics
import statistics
import argparse

from neural_verification import MLP, MLPConfig


parser = argparse.ArgumentParser(description='Input the filename of an MLP model, retrain this model.')
parser.add_argument('fname', type=str, help='The name of the model file')
args = parser.parse_args()
fname = args.fname


# Load the weights and biases from fname
original_weights = torch.load(fname, map_location=torch.device('cpu'))
if tuple(sorted(original_weights.keys())) == tuple(sorted(['mlp.0.weight', 'mlp.0.bias', 'mlp.2.weight', 'mlp.2.bias'])):
    key1_key2_pairs = [('mlp.0.weight', 'linears.0.weight'), ('mlp.0.bias', 'linears.0.bias'), ('mlp.2.weight', 'linears.1.weight'), ('mlp.2.bias', 'linears.1.bias')]
    original_weights = {key2:original_weights[key1] for (key1, key2) in key1_key2_pairs}

prefix = 'linears.'
original_shape = [original_weights[prefix + '0.bias'].shape[0]] + [original_weights[prefix + str(i) + '.bias'].shape[0] for i in range(int(len(original_weights)//2))]
weights = [original_weights[prefix + str(i) + '.weight'].numpy() for i in range(int(len(original_weights)//2))]
biases = [original_weights[prefix + str(i) + '.bias'].numpy() for i in range(int(len(original_weights)//2))]

shp = [weights.shape[1]] + [bias.shape[0] for bias in biases]
depth = len(shp)-1
width = max(shp[1:-1])
in_dim = shp[0]
out_dim = shp[-1]
model = MLP(in_dim=in_dim, out_dim=out_dim, width=width, depth=depth)
linear_list = []
for i in range(depth):
    linear_list.append(nn.Linear(shp[i], shp[i+1]))
model.linears = nn.ModuleList(linear_list)
model.shp = shp
model.load_state_dict(original_weights)

torch.save(model.state_dict(), fname[:-3] + "_standardized.pt")

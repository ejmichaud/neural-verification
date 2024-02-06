import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import yaml
import os
import itertools

from neural_verification import MLP, MLPConfig
from neural_verification import (
    GeneralRNNConfig,
    GeneralRNN,
    cycle,
    FastTensorDataLoader
)

def prepare_xy(x, y, vectorize_input, input_dim, dtype, loss_fn):  # Copied from scripts/rnn_train.py
    # HANDLE X CONVERSIONS
    if vectorize_input:
        x = x.to(torch.int64)
        x = F.one_hot(x, num_classes=input_dim)
    if len(x.shape) == 2: # (batch_size, seq_len)
        x = x.unsqueeze(2) # (batch_size, seq_len, 1)
    x = x.to(dtype)
    # HANDLE Y CONVERSIONS
    if loss_fn == "cross_entropy":
        y = y.to(torch.int64)
    else:
        if len(y.shape) == 2: # (batch_size, seq_len)
            y = y.unsqueeze(2) # (batch_size, seq_len, 1)
        y = y.to(dtype)
    return x, y

parser = argparse.ArgumentParser(description='Input the filename of an MLP model to measure the simplicity of.')
parser.add_argument('task', type=str, help='The name of the task')
parser.add_argument('fname', type=str, help='The name of the model file')
args = parser.parse_args()
fname = args.fname
task = args.task

# Read search.yaml
with open("../search.yaml", 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

task_config = config[task]

task_args = task_config['args']
vectorize_input = task_args['vectorize_input']
loss_fn = task_args['loss_fn']  # 'cross_entropy' for example
task_path = fname
if not os.path.exists(task_path):
    raise ValueError("trained network not found: " + task_path)

print("loaded")
if torch.cuda.is_available():
    weights = torch.load(task_path)
else:
    weights = torch.load(task_path, map_location=torch.device('cpu'))

# Get the RNN shape
hidden_mlp_depth = 0
output_mlp_depth = 0
for key in weights.keys():
    if key[:4] == 'hmlp' and key[-6:] == 'weight':
        hidden_mlp_depth += 1
    if key[:4] == 'ymlp' and key[-6:] == 'weight':
        output_mlp_depth += 1
hidden_dim = weights['hmlp.mlp.' + str(hidden_mlp_depth*2-2) + '.weight'].shape[0]
output_dim = weights['ymlp.mlp.' + str(output_mlp_depth*2-2) + '.weight'].shape[0]
input_dim = weights['hmlp.mlp.0.weight'].shape[1] - hidden_dim
if hidden_mlp_depth >= 2:
    hidden_mlp_width = weights['hmlp.mlp.0.weight'].shape[0]
else:
    hidden_mlp_width = hidden_dim
if output_mlp_depth >= 2:
    output_mlp_width = weights['ymlp.mlp.0.weight'].shape[0]
else:
    output_mlp_width = output_dim
activation = getattr(torch.nn, task_args['activation'])

config = GeneralRNNConfig(input_dim, output_dim, hidden_dim, hidden_mlp_depth, hidden_mlp_width, output_mlp_depth, output_mlp_width, activation)

# Get the RNN weight
rnn = GeneralRNN(config, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
rnn.load_state_dict(weights)
rnn.eval()

# Count up the weights and neurons
epsilon_threshold = 0.01
with torch.no_grad():
    hmlp_weight_count = 0
    hmlp_neuron_count = 0
    ymlp_weight_count = 0
    ymlp_neuron_count = 0
    weight_norm = 0
    for i, (layer1, layer2) in enumerate(zip(rnn.hmlp.mlp[:-2:2], rnn.hmlp.mlp[2::2])):
        hmlp_neuron_count += np.sum(np.logical_and(np.any(np.abs(layer1.weight.numpy()) > epsilon_threshold, axis=1), np.any(np.abs(layer2.weight.numpy()) > epsilon_threshold, axis=0)).astype(np.int32))
    for i, (layer1, layer2) in enumerate(zip(rnn.ymlp.mlp[:-2:2], rnn.ymlp.mlp[2::2])):
        ymlp_neuron_count += np.sum(np.logical_and(np.any(np.abs(layer1.weight.numpy()) > epsilon_threshold, axis=1), np.any(np.abs(layer2.weight.numpy()) > epsilon_threshold, axis=0)).astype(np.int32))
    for i, layer in enumerate(rnn.hmlp.mlp[::2]):
        hmlp_weight_count += np.sum((np.abs(layer.weight.numpy()) > epsilon_threshold).astype(np.int32))
        weight_norm += np.sum(layer.weight.numpy()**2)
    for i, layer in enumerate(rnn.ymlp.mlp[::2]):
        ymlp_weight_count += np.sum((np.abs(layer.weight.numpy()) > epsilon_threshold).astype(np.int32))
        weight_norm += np.sum(layer.weight.numpy()**2)
    weight_count = hmlp_weight_count + ymlp_weight_count
    neuron_count = hmlp_neuron_count + ymlp_neuron_count
    weight_norm = np.sqrt(weight_norm)

    hidden_writes = np.any(np.abs(rnn.hmlp.mlp[-1].weight.numpy()) > epsilon_threshold, axis=1)
    hidden_hmlp_reads = np.any(np.abs(rnn.hmlp.mlp[0].weight.numpy()[:,:hidden_dim]) > epsilon_threshold, axis=0)
    hidden_ymlp_reads = np.any(np.abs(rnn.ymlp.mlp[0].weight.numpy()) > epsilon_threshold, axis=0)
    hidden_dim = np.sum(np.logical_and(hidden_writes, np.logical_or(hidden_hmlp_reads, hidden_ymlp_reads)))


print("In dim: " + str(neuron_count))
print("Out dim: " + str(neuron_count))
print("h width: " + str(hidden_mlp_width))
print("h depth: " + str(hidden_mlp_depth))
print("y width: " + str(output_mlp_width))
print("y depth: " + str(output_mlp_depth))
print("Neurons: " + str(neuron_count))
print("Weights: " + str(weight_count))
print("Weight norm: " + str(weight_norm))
print("Hidden dim: " + str(hidden_dim))

def betterprint(X):
    print(goodrepr(X))
def goodrepr(X):
    if isinstance(X, dict):
        return "{" + ",\n".join([key + ":\n" + goodrepr(val) for key, val in X.items()]) + "\n}"
    if type(X) == list:
        return "[" + "\n".join([goodrepr(x) for x in X]) + "]"
    elif isinstance(X, np.ndarray):
        return np.array2string(X, max_line_width=100, precision=4, suppress_small=True)
    elif torch.is_tensor(X):
        return goodrepr(X.numpy())
    else:
        return str(X)
#        print(type(X))
#        raise ValueError

print("")
print("Weights:")
betterprint(weights)
print("")
print("")


cumulative_product = torch.eye(hidden_dim+input_dim)
for i in range((len(rnn.hmlp.mlp)+1)//2):
    cumulative_product = torch.matmul(rnn.hmlp.mlp[2*i].weight, cumulative_product)
with torch.no_grad():
    W = cumulative_product.numpy()

betterprint(W)

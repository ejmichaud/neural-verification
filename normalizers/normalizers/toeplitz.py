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

parser = argparse.ArgumentParser(description='Input the filename of an RNN, simplify this model.')
parser.add_argument('task', type=str, help='The name of the task')
parser.add_argument('fname', type=str, help='The name of the model file')
parser.add_argument('modified_fname', type=str, help='The name of the new model file to be made')
args = parser.parse_args()
task = args.task
fname = args.fname
modified_fname = args.modified_fname


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
    hidden_mlp_width = 1
if output_mlp_depth >= 2:
    output_mlp_width = weights['ymlp.mlp.0.weight'].shape[0]
else:
    output_mlp_width = 1
activation = getattr(torch.nn, task_args['activation'])

config = GeneralRNNConfig(input_dim, output_dim, hidden_dim, hidden_mlp_depth, hidden_mlp_width, output_mlp_depth, output_mlp_width, activation)


# Get the RNN weight
rnn = GeneralRNN(config, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
rnn.load_state_dict(weights)
rnn.eval()

cumulative_product = torch.eye(hidden_dim+input_dim)
for i in range((len(rnn.hmlp.mlp)+1)//2):
    cumulative_product = torch.matmul(rnn.hmlp.mlp[2*i].weight, cumulative_product)
with torch.no_grad():
    W = cumulative_product.numpy()[:,:hidden_dim]
    X = cumulative_product.numpy()[:,hidden_dim:]
    X_bias = rnn.hmlp.mlp[-1].bias.numpy()

def test_jordan_canonical(W):
    assert W.shape[0] == W.shape[1], "Matrix must be square."
    hidden_dim = W.shape[0]
    blocks = [1]
    max_tolerance = 0
    for i in range(hidden_dim):
        if i > 0:
            max_tolerance = np.maximum(max_tolerance, np.max(np.abs(W[i,:i])))
        if i < hidden_dim - 2:
            max_tolerance = np.maximum(max_tolerance, np.max(np.abs(W[i,i+2:])))
        if i < hidden_dim-1:
            one = bool(W[i,i+1] > 0.5)
            max_tolerance = np.maximum(max_tolerance, np.abs(int(one)-W[i,i+1]))
            if one:
                blocks[-1] += 1
            if not one:
                blocks.append(1)
    partitions = np.cumsum([0] + blocks).tolist()

    eigvals = []
    for start, end in zip(partitions[:-1], partitions[1:]):
        eigvals.append(np.mean(np.diagonal(W)[start:end]))
        max_tolerance = np.maximum(max_tolerance, np.max(np.abs(np.diagonal(W)[start:end]-eigvals[-1])))
    
    min_intolerance = np.inf
    for i in range(len(eigvals)):
        for j in range(i):
            min_intolerance = np.minimum(min_intolerance, np.abs(eigvals[i]-eigvals[j]))

    assert max_tolerance < min_intolerance, "Matrix not in jordan canonical form."
    assert max_tolerance < 1/2, "Matrix not in jordan canonical form."

    return np.nextafter(max_tolerance, 1), blocks, eigvals
    
try:
    epsilon, block_structure, eigvals = test_jordan_canonical(W)
    is_in_jordan_canonical = True
except AssertionError:
    is_in_jordan_canonical = False

print(W)
if is_in_jordan_canonical:
    transform = np.zeros((hidden_dim, hidden_dim), dtype=W.dtype)
    block_splits = np.cumsum([0] + block_structure)
    print(block_splits, X, X_bias)
    for start, end in zip(block_splits[:-1], block_splits[1:]):
        while end > start:
            size = end-start
            X_block = np.concatenate([X[start:end,:], X_bias[start:end,np.newaxis]], axis=1)
            input_choice = np.argmax(np.abs(X_block[-1,:]))
            if np.abs(X_block[-1,input_choice]) > epsilon and np.abs(X_block[-1,input_choice]) > 0.0001:  # Last neuron of jordan block is not dead and has inputs; Toeplitz will be invertible.
                block_transform = np.zeros((size, size), dtype=X_block.dtype)  # construct an upper triangular Toeplitz matrix. Upper triangular Toeplitz matrices (Jordan blocks included) always commute with each other.
                i, j = np.indices((size, size))
                for diagonal in range(size):
                    block_transform[i+diagonal==j] = X_block[-diagonal-1,input_choice]
                transform[start:end,start:end] = block_transform
                print(transform[start:end,start:end])
                break
            else:
                transform[end-1,end-1] = 1
                print("A", end)
                end -= 1
        print(X_block, transform[start:end,start:end])
else:
    transform = np.eye(hidden_dim)

try:
    transform_in = np.linalg.inv(transform)
except:
    print(transform)
    raise ValueError
transform_out = transform

transform_in = transform_in.astype(np.cdouble)
transform_out = transform_out.astype(np.cdouble)

with torch.no_grad():
    weight0 = rnn.hmlp.mlp[-1].weight.type(torch.cdouble)
    weight1 = rnn.hmlp.mlp[-1].bias.type(torch.cdouble)
    weight2 = rnn.hmlp.mlp[0].weight.type(torch.cdouble)
    weight3 = rnn.ymlp.mlp[0].weight.type(torch.cdouble)

    weight0 = torch.matmul(torch.tensor(transform_in), weight0)
    if len(rnn.hmlp.mlp) == 1:
        weight2 = weight0
    weight1 = torch.matmul(torch.tensor(transform_in), weight1)
    weight2 = torch.cat([torch.matmul(weight2[:,:hidden_dim], torch.tensor(transform_out)), weight2[:,hidden_dim:]], dim=1)
    if len(rnn.hmlp.mlp) == 1:
        weight0 = weight2
    weight3 = torch.matmul(weight3, torch.tensor(transform_out))

    rnn.hmlp.mlp[-1].weight = nn.Parameter(weight0.real.float())
    rnn.hmlp.mlp[-1].bias = nn.Parameter(weight1.real.float())
    rnn.hmlp.mlp[0].weight = nn.Parameter(weight2.real.float())
    rnn.ymlp.mlp[0].weight = nn.Parameter(weight3.real.float())

# Put the new weights and biases into modified_model.pt
torch.save(rnn.state_dict(), modified_fname)

print('Transformed RNN hidden space using matrix: ' + str(transform_in))

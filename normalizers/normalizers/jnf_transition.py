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
cumulative_product = cumulative_product[:,:hidden_dim]
with torch.no_grad():
    eigvals, eigvects = np.linalg.eig(cumulative_product.numpy())
if hidden_dim <= 5:
    permutations = list(map(np.array, itertools.permutations(list(range(hidden_dim)))))
else:
    permutations = [np.random.permutation(hidden_dim) for i in range(120)]
with torch.no_grad():
    best_V = None
    best_det = 0
    for permutation in permutations:
        jnf = np.diag(eigvals[permutation])
        i, j = np.indices(cumulative_product.shape)
        jnf[i+1==j] = 1
        S, V = np.linalg.eig(jnf)  # Vdiag(S)V^{-1}=jnf
        det = np.abs(np.linalg.det(V))
        if det > best_det:
            best_det = det
            best_V = V
    transform = torch.tensor(eigvects.dot(np.linalg.inv(best_V)))
    inv_transform = torch.tensor(best_V.dot(np.linalg.inv(eigvects)))
    if transform.dtype == torch.cfloat:
        transform = transform.real
    if inv_transform.dtype == torch.cfloat:
        inv_transform = inv_transform.real


with torch.no_grad():
    rnn.hmlp.mlp[-1].weight = nn.Parameter(torch.matmul(inv_transform, rnn.hmlp.mlp[-1].weight))
    rnn.hmlp.mlp[-1].bias = nn.Parameter(torch.matmul(inv_transform, rnn.hmlp.mlp[-1].bias))
    rnn.hmlp.mlp[0].weight = nn.Parameter(torch.cat([torch.matmul(rnn.hmlp.mlp[0].weight[:,:hidden_dim], transform), rnn.hmlp.mlp[0].weight[:,hidden_dim:]], dim=1))
    rnn.ymlp.mlp[0].weight = nn.Parameter(torch.matmul(rnn.ymlp.mlp[0].weight, transform))

# Put the new weights and biases into modified_model.pt
torch.save(rnn.state_dict(), modified_fname)

print('Transformed RNN hidden space using matrix: ' + str(transform))


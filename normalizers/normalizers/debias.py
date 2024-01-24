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
parser.add_argument('-e', '--epsilon', type=float, help='Error tolerance', default=0.1)
args = parser.parse_args()
task = args.task
fname = args.fname
modified_fname = args.modified_fname
epsilon = args.epsilon


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

with torch.no_grad():
    h_write = rnn.hmlp.mlp[-1].weight.numpy()
    h_write_bias = rnn.hmlp.mlp[-1].bias.numpy()
    h_read = rnn.hmlp.mlp[0].weight[:,:hidden_dim].numpy()
    h_read_bias = rnn.hmlp.mlp[0].bias.numpy()
    y_read = rnn.ymlp.mlp[0].weight.numpy()
    y_read_bias = rnn.ymlp.mlp[0].bias.numpy()

I = np.eye(hidden_dim)
U, S, Vh = np.linalg.svd(h_read)
rank = np.sum((S > epsilon).astype(np.int32))
nullspace = Vh[rank:,:].T
projector = nullspace.dot(nullspace.T)

with torch.no_grad():
    rnn.hmlp.mlp[-1].bias = nn.Parameter(torch.tensor((I-projector).dot(h_write_bias)).type(rnn.hmlp.mlp[-1].bias.dtype))
    rnn.ymlp.mlp[0].bias = nn.Parameter(torch.tensor(y_read_bias + y_read.dot(projector).dot(h_write_bias)).type(rnn.ymlp.mlp[0].bias.dtype))

# Put the new weights and biases into modified_model.pt
torch.save(rnn.state_dict(), modified_fname)

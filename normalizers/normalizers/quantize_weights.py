import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import yaml
import os

from neural_verification import MLP, MLPConfig
from neural_verification import (
    GeneralRNNConfig,
    GeneralRNN,
    cycle,
    FastTensorDataLoader
)

parser = argparse.ArgumentParser(description='Input the filename of an RNN, simplify this model.')
parser.add_argument('task', type=str, help='The name of the task')
parser.add_argument('fname', type=str, help='The name of the model file')
parser.add_argument('modified_fname', type=str, help='The name of the new model file to be made')
# This threshold tells the program how dead a neuron must be to be considered "dead". For any given neuron, it is the product of L2 norms of the vectors of incoming and outgoing weights.
parser.add_argument('-t', '--prune_threshold', type=float, help='Magnitude threshold to prune neurons', default=0.01)
args = parser.parse_args()
task = args.task
fname = args.fname
modified_fname = args.modified_fname
prune_threshold = args.prune_threshold


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


# Prune the hidden dimension
with torch.no_grad():
    hmlp_weights = [layer.weight.numpy() for layer in rnn.hmlp.mlp[::2]]
    hmlp_biases = [layer.bias.numpy() for layer in rnn.hmlp.mlp[::2]]
    ymlp_weights = [layer.weight.numpy() for layer in rnn.ymlp.mlp[::2]]
    ymlp_biases = [layer.bias.numpy() for layer in rnn.ymlp.mlp[::2]]

    n_weights = 0
    n_quantized_weights = 0
    for tensor_list in (hmlp_weights, hmlp_biases, ymlp_weights, ymlp_biases):
        for i in range(len(tensor_list)):
            n_weights = n_weights + np.sum(np.ones_like(tensor_list[i]))
            n_quantized_weights = n_quantized_weights + np.sum((np.abs(np.round(tensor_list[i]) - tensor_list[i]) < prune_threshold).astype(np.int32))
            tensor_list[i] = np.where(np.abs(np.round(tensor_list[i]) - tensor_list[i]) < prune_threshold, np.round(tensor_list[i]), tensor_list[i])

    hmlp_weights = {'hmlp.mlp.'+str(i*2)+'.weight':torch.from_numpy(hmlp_weights[i]) for i in range(len(hmlp_weights))}
    hmlp_biases = {'hmlp.mlp.'+str(i*2)+'.bias':torch.from_numpy(hmlp_biases[i]) for i in range(len(hmlp_biases))}
    ymlp_weights = {'ymlp.mlp.'+str(i*2)+'.weight':torch.from_numpy(ymlp_weights[i]) for i in range(len(ymlp_weights))}
    ymlp_biases = {'ymlp.mlp.'+str(i*2)+'.bias':torch.from_numpy(ymlp_biases[i]) for i in range(len(ymlp_biases))}

# Put the new weights and biases into modified_model.pt
new_weights = {**hmlp_weights, **hmlp_biases, **ymlp_weights, **ymlp_biases}
rnn.load_state_dict(new_weights)
torch.save(rnn.state_dict(), modified_fname)

print(str(n_quantized_weights) + ' out of ' + str(n_weights) + ' weights quantized.')

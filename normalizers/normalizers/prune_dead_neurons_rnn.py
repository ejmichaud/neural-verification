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
parser.add_argument('-t', '--prune_threshold', type=float, help='Magnitude threshold to prune neurons', default=0.2)  # 0.32 prunes some neurons in a random network
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


def figure_out_which_neurons_to_prune(weights):
    """
    A function to determine which neurons we should prune if given the weights. ie. try to detect neurons which do nothing.

    weights: a length N list of [out_dim, in_dim] weight tensors for the MLP.
    
    returns: a length N-1 list of length n_neurons lists of booleans - True if that neuron in that hidden layer should be removed, false otherwise.
    """
    neurons_to_prune = []
    for layer in range(len(weights)-1):
        neurons_to_prune.append([])
        for neuron_num in range(weights[layer].shape[0]):
            in_weights = weights[layer][neuron_num,:]
            out_weights = weights[layer+1][:,neuron_num]
            in_norm = np.sqrt(np.sum(in_weights**2))
            out_norm = np.sqrt(np.sum(out_weights**2))
            max_jacobian = in_norm*out_norm
            neurons_to_prune[layer].append(max_jacobian < prune_threshold)
    return neurons_to_prune

def prune_neurons(weights, biases, neurons_to_prune):
    """
    A function that cuts out all the neurons that should be pruned, and modifies the weights and biases in place. prune_neurons determines which weights and biases to cut out.
    """
    for layer in range(len(weights)-1):
        for neuron_num in reversed(range(weights[layer].shape[0])):
            if neurons_to_prune[layer][neuron_num]:
                biases[layer+1] = biases[layer+1] + activation()(torch.from_numpy(np.array(biases[layer][neuron_num]))).numpy()*weights[layer+1][:,neuron_num]
                biases[layer] = np.concatenate([biases[layer][:neuron_num], biases[layer][neuron_num+1:]], axis=0)
                weights[layer] = np.concatenate([weights[layer][:neuron_num,:], weights[layer][neuron_num+1:,:]], axis=0)
                weights[layer+1] = np.concatenate([weights[layer+1][:,:neuron_num], weights[layer+1][:,neuron_num+1:]], axis=1)

def expand_to_shape(weights, biases, shp):
    """
    This function takes in weights and biases for a neural network of a smaller shape and expands the weights and biases with dead neurons until it fills a given shape shp.
    This function modifies weights and biases in place.
    """
    for layer in range(len(weights)-1):
        if weights[layer].shape[0] < shp[layer+1]:
            weights[layer] = np.concatenate([weights[layer], np.zeros([shp[layer+1]-weights[layer].shape[0], weights[layer].shape[1]], dtype=weights[layer].dtype)], axis=0)
            weights[layer+1] = np.concatenate([weights[layer+1], np.zeros([weights[layer+1].shape[0], shp[layer+1]-weights[layer+1].shape[1]], dtype=weights[layer+1].dtype)], axis=1)
            biases[layer] = np.concatenate([biases[layer], np.zeros([shp[layer+1]-biases[layer].shape[0]], dtype=biases[layer].dtype)], axis=0)

# Prune the dead neurons
with torch.no_grad():
    new_widths = []
    new_state_dicts = []
    for mlp, prefix in [(rnn.hmlp.mlp, 'hmlp'), (rnn.ymlp.mlp, 'ymlp')]:
        weights = [layer.weight.numpy() for layer in mlp[::2]]
        biases = [layer.bias.numpy() for layer in mlp[::2]]

        # Figure out which neurons are dead and should be pruned
        neurons_to_prune = figure_out_which_neurons_to_prune(weights)

        # Prune dead neurons
        prune_neurons(weights, biases, neurons_to_prune)
        new_width = max([1] + [weight.shape[0] for weight in weights[:-1]])
        new_shape = [weights[0].shape[1]] + [new_width for i in range(len(weights)-1)] + [weights[-1].shape[0]]
        new_widths.append(new_width)
        expand_to_shape(weights, biases, new_shape)

        # Put the new weights and biases into a data structure that can be loaded into a pytorch model
        weights = {prefix+'.mlp.'+str(i*2)+'.weight':torch.from_numpy(weights[i]) for i in range(len(weights))}
        biases = {prefix+'.mlp.'+str(i*2)+'.bias':torch.from_numpy(biases[i]) for i in range(len(biases))}
        new_state_dicts.append({**weights, **biases})
new_weights = {**new_state_dicts[0], **new_state_dicts[1]}

# Put the new weights and biases into modified_model.pt
new_config = GeneralRNNConfig(input_dim, output_dim, hidden_dim, hidden_mlp_depth, new_widths[0], output_mlp_depth, new_widths[1], activation)
new_rnn = GeneralRNN(new_config, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
new_rnn.load_state_dict(new_weights)
torch.save(new_rnn.state_dict(), modified_fname)

print('Original widths [h network, y network]: [' + str(hidden_mlp_width) + ', ' + str(output_mlp_width) + ']. New shape: ' + str(new_widths))

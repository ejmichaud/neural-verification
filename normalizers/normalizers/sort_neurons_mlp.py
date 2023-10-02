import torch
import torch.nn as nn
import numpy as np
import argparse

from neural_verification import MLP


parser = argparse.ArgumentParser(description='Input the filename of an MLP model, simplify this model.')
parser.add_argument('fname', type=str, help='The name of the model file')
parser.add_argument('modified_fname', type=str, help='The name of the new model file to be made')
args = parser.parse_args()
fname = args.fname
modified_fname = args.modified_fname


N = 10000 # number of samples


def sort_neurons(weights, biases):
    """
    A function to sort the neurons by their strength/influence. This is defined as the sum of absolute values of incoming and outgoing weights.

    weights: a length N list of [out_dim, in_dim] weight tensors for the MLP.
    """
    for layer, (in_weights, out_weights, bias) in enumerate(zip(weights[:-1], weights[1:], biases[:-1])):
        order = np.argsort(-(np.sum(np.abs(in_weights), axis=1) + np.abs(bias) + np.sum(np.abs(out_weights), axis=0)))  # order
        weights[layer] = weights[layer][order]
        biases[layer] = biases[layer][order]
        weights[layer+1] = weights[layer+1].T[order].T


# Load the weights and biases from fname
original_weights = torch.load(fname, map_location=torch.device('cpu'))
#key1_key2_pairs = [('mlp.0.weight', 'linears.0.weight'), ('mlp.0.bias', 'linears.0.bias'), ('mlp.2.weight', 'linears.1.weight'), ('mlp.2.bias', 'linears.1.bias')]
#original_weights = {key2:original_weights[key1] for (key1, key2) in key1_key2_pairs}
prefix = 'linears.'
original_shape = [original_weights[prefix + '0.bias'].shape[0]] + [original_weights[prefix + str(i) + '.bias'].shape[0] for i in range(int(len(original_weights)//2))]
weights = [original_weights[prefix + str(i) + '.weight'].numpy() for i in range(int(len(original_weights)//2))]
biases = [original_weights[prefix + str(i) + '.bias'].numpy() for i in range(int(len(original_weights)//2))]

# Sort neurons
sort_neurons(weights, biases)

# Put the new weights and biases into a data structure that can be loaded into a pytorch model
weights = {prefix+str(i)+'.weight':torch.from_numpy(weights[i]) for i in range(len(weights))}
biases = {prefix+str(i)+'.bias':torch.from_numpy(biases[i]) for i in range(len(biases))}
new_weights = {**weights, **biases}

# Put the new weights and biases into modified_model.pt
depth=int(len(new_weights)//2)
width=new_weights[prefix + '0.weight'].shape[0]
in_dim=new_weights[prefix + '0.weight'].shape[1]
out_dim=new_weights[prefix +  str(depth-1) + '.weight'].shape[0]
model = MLP(in_dim=in_dim, out_dim=out_dim, width=width, depth=depth)
model.load_state_dict(new_weights)
model.shp = [new_weights[prefix + '0.bias'].shape[0]] + [new_weights[prefix + str(i) + '.bias'].shape[0] for i in range(int(len(new_weights)//2))]
torch.save(model.state_dict(), modified_fname)

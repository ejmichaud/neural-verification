import torch
import torch.nn as nn
import numpy as np
import argparse

from neural_verification import MLP


##### THIS PROGRAM IS ONLY GOOD FOR SIMPLIFYING RELU MLPS. DO NOT USE IT ON MLPS OF ANY OTHER ACTIVATION FUNCTION.


parser = argparse.ArgumentParser(description='Input the filename of an MLP model, simplify this model. NOTE: ONLY USE THIS ON ReLU MLPs!!!')
parser.add_argument('fname', type=str, help='The name of the model file')
parser.add_argument('modified_fname', type=str, help='The name of the new model file to be made')
args = parser.parse_args()
fname = args.fname
modified_fname = args.modified_fname


def normalize_weights(weights, biases):
    """
    A function to normalize the vector of weights going into each neuron.

    weights: a length N list of [out_dim, in_dim] weight tensors for the MLP.
    """
    for layer in range(len(weights)-1):
        magnitudes = np.sqrt(np.mean(weights[layer]**2, axis=1))
        weights[layer] = weights[layer] / magnitudes[:,np.newaxis]
        biases[layer] = biases[layer] / magnitudes
        weights[layer+1] = weights[layer+1] * magnitudes


# Load the weights and biases from fname
original_weights = torch.load(fname, map_location=torch.device('cpu'))
prefix = 'linears.'
original_shape = [original_weights[prefix + '0.weight'].shape[1]] + [original_weights[prefix + str(i) + '.bias'].shape[0] for i in range(int(len(original_weights)//2))]
weights = [original_weights[prefix + str(i) + '.weight'].numpy() for i in range(int(len(original_weights)//2))]
biases = [original_weights[prefix + str(i) + '.bias'].numpy() for i in range(int(len(original_weights)//2))]

# Sort neurons
normalize_weights(weights, biases)

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

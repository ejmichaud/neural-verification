import torch
import torch.nn as nn
import numpy as np

from neural_verification import RNN


# This threshold tells the program how dead a neuron must be to be considered "dead". For any given neuron, it is 1.1 times the product of L2 norms of the vectors of incoming and outgoing weights.
prune_threshold = 0.32

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
            max_jacobian = in_norm*out_norm*1.1  # the true number is the maximum slope of the SiLU function, which is a bit less than 1.1
            neurons_to_prune[layer].append(max_jacobian < prune_threshold)
    return neurons_to_prune

def prune_neurons(weights, biases, neurons_to_prune):
    """
    A function that cuts out all the neurons that should be pruned, and modifies the weights and biases in place. prune_neurons determines which weights and biases to cut out.
    """
    for layer in range(len(weights)-1):
        for neuron_num in reversed(range(weights[layer].shape[0])):
            if neurons_to_prune[layer][neuron_num]:
                biases[layer+1] = biases[layer+1] + torch.nn.SiLU()(torch.from_numpy(np.array(biases[layer][neuron_num]))).numpy()*weights[layer+1][:,neuron_num]
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



# Load the weights and biases from sample_rnn.pt
original_weights = torch.load('sample_rnn.pt', map_location=torch.device('cpu'))
input_dim = original_weights['linears.0.weight'].shape[1]
hidden_dim = original_weights['linears.0.weight'].shape[0]
output_dim = original_weights['linears.2.weight'].shape[0]
Wh = original_weights['linears.0.weight'].numpy()
Wx = original_weights['linears.1.weight'].numpy()
Wy = original_weights['linears.2.weight'].numpy()
bh = original_weights['linears.0.bias'].numpy()
bx = original_weights['linears.1.bias'].numpy()
by = original_weights['linears.2.bias'].numpy()

# Figure out which neurons are dead and should be pruned
neurons_to_prune = figure_out_which_neurons_to_prune(Wh, Wx, Wy)

# Prune dead neurons
prune_neurons(Wh, Wx, Wy, bh, bx, by, neurons_to_prune)

# Fill the dead neurons with zeroed out neurons
expand_to_shape(Wh, Wx, Wy, bh, bx, by, original_shape)

# Count the neurons which have been pruned and how many remain
n_pruned_neurons = sum(list(map(int, neurons_to_prune)))
n_unpruned_neurons = len(neurons_to_prune) - n_pruned_neurons
new_hidden_dim = n_unpruned_neurons

# Put the new weights and biases into a data structure that can be loaded into a pytorch model
weights = {'linears.0.weight':Wh, 'linears.1.weight':Wx, 'linears.2.weight':Wy}
biases = {'linears.0.bias':Wh, 'linears.1.bias':Wx, 'linears.2.bias':Wy}
new_weights = {**weights, **biases}

# Put the new weights and biases into modified_rnn.pt
model = RNN(hidden_dim, input_dim=input_dim, output_dim=output_dim)
model.load_state_dict(new_weights)
torch.save(model.state_dict(), 'modified_rnn.pt')

print(str(n_pruned_neurons) + ' hidden neurons pruned, ' + str(n_unpruned_neurons) + ' hidden neurons left.)

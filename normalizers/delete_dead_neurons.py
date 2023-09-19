import torch
import torch.nn as nn
import numpy as np

from neural_verification import MLP


prune_threshold = 0.32

def figure_out_which_neurons_to_prune(weights):
    """
    weights: a length N list of [out_dim, in_dim] weight tensors for the MLP.
    
    returns: a length N-1 list of length n_neurons lists of booleans - True if that neuron in the ith hidden layer should be removed, false otherwise.
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
    Cut out all the neurons that prune_neurons says to cut out.
    This function modifies weights and biases in place.
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
    This function modifies weights and biases in place.
    """
    for layer in range(len(weights)-1):
        if weights[layer].shape[0] < shp[layer+1]:
            weights[layer] = np.concatenate([weights[layer], np.zeros([shp[layer+1]-weights[layer].shape[0], weights[layer].shape[1]], dtype=weights[layer].dtype)], axis=0)
            weights[layer+1] = np.concatenate([weights[layer+1], np.zeros([weights[layer+1].shape[0], shp[layer+1]-weights[layer+1].shape[1]], dtype=weights[layer+1].dtype)], axis=1)
            biases[layer] = np.concatenate([biases[layer], np.zeros([shp[layer+1]-biases[layer].shape[0]], dtype=biases[layer].dtype)], axis=0)



original_weights = torch.load('sample_model.pt', map_location=torch.device('cpu'))
original_shape = [original_weights['linears.0.bias'].shape[0]] + [original_weights['linears.' + str(i) + '.bias'].shape[0] for i in range(int(len(original_weights)//2))]
weights = [original_weights['linears.' + str(i) + '.weight'].numpy() for i in range(int(len(original_weights)//2))]
biases = [original_weights['linears.' + str(i) + '.bias'].numpy() for i in range(int(len(original_weights)//2))]

neurons_to_prune = figure_out_which_neurons_to_prune(weights)
prune_neurons(weights, biases, neurons_to_prune)
expand_to_shape(weights, biases, original_shape)
n_pruned_neurons = sum([sum(list(map(int, layer))) for layer in neurons_to_prune])
new_shape = [original_shape[0]] + [sum([1-int(x) for x in layer]) for layer in neurons_to_prune] + [original_shape[-1]]
n_unpruned_neurons = sum(new_shape[1:-1])

weights = {'linears.'+str(i)+'.weight':torch.from_numpy(weights[i]) for i in range(len(weights))}
biases = {'linears.'+str(i)+'.bias':torch.from_numpy(biases[i]) for i in range(len(biases))}
new_weights = {**weights, **biases}

depth=int(len(new_weights)//2)
in_dim=new_weights['linears.0.weight'].shape[1]
out_dim=new_weights['linears.' + str(depth-1) + '.weight'].shape[0]
width=new_weights['linears.0.weight'].shape[0]

model = MLP(in_dim=in_dim, out_dim=out_dim, width=width, depth=depth)
model.load_state_dict(new_weights)
model.shp = [new_weights['linears.0.bias'].shape[0]] + [new_weights['linears.' + str(i) + '.bias'].shape[0] for i in range(int(len(new_weights)//2))]
torch.save(model.state_dict(), 'modified_model.pt')
print(str(n_pruned_neurons) + ' hidden neurons pruned, ' + str(n_unpruned_neurons) + ' hidden neurons left. Original shape: ' + str(original_shape) + '. New shape: ' + str(new_shape))

import torch
import torch.nn as nn
import numpy as np

from neural_verification import MLP


epsilon = 0.001
prune_threshold = 0.55

def try_to_combine_neurons(weights, biases):
    """
    weights: a length N list of [out_dim, in_dim] weight tensors for the MLP.

    This function finds the first instance of two neurons that compute identical values. They both get removed, an average neuron gets put at the end, and then a dead neuron gets put at the end.
    
    This function modifies the weights and biases in place.
    """
    neurons_to_prune = []
    for layer in range(len(weights)-1):
        neurons_to_prune.append([])
        for neuron_num_1 in range(weights[layer].shape[0]):
            in_weights_1 = weights[layer][neuron_num_1,:]
            out_weights_1 = weights[layer+1][:,neuron_num_1]
            for neuron_num_2 in range(neuron_num_1+1, weights[layer].shape[0]):
                in_weights_2 = weights[layer][neuron_num_2,:]
                out_weights_2 = weights[layer+1][:,neuron_num_2]
                norm_2 = np.sqrt(np.sum(in_weights_2**2) + biases[layer][neuron_num_2]**2)
                diff = in_weights_1-in_weights_2
                norm = np.sqrt(np.sum(diff**2) + (biases[layer][neuron_num_1]-biases[layer][neuron_num_2])**2)
                if norm < prune_threshold and norm_2 > epsilon:
                    avg_weight = (in_weights_1 + in_weights_2) / 2
                    avg_bias = (biases[layer][neuron_num_1] + biases[layer][neuron_num_2]) / 2
                    total_output = out_weights_1 + out_weights_2
                    weights[layer] = np.concatenate([weights[layer][:neuron_num_1,:], weights[layer][neuron_num_1+1:neuron_num_2,:], weights[layer][neuron_num_2+1:,:], avg_weight[np.newaxis,:], np.zeros([1, weights[layer].shape[1]])], axis=0)
                    weights[layer+1] = np.concatenate([weights[layer+1][:,:neuron_num_1], weights[layer+1][:,neuron_num_1+1:neuron_num_2], weights[layer+1][:,neuron_num_2+1:], total_output[:,np.newaxis], np.zeros([weights[layer+1].shape[0], 1])], axis=1)
                    biases[layer] = np.concatenate([biases[layer][:neuron_num_1], biases[layer][neuron_num_1+1:neuron_num_2], biases[layer][neuron_num_2+1:], np.array([avg_bias, 0])], axis=0)
                    return (layer, neuron_num_1, neuron_num_2, norm)
    return False

original_weights = torch.load('sample_model.pt', map_location=torch.device('cpu'))
original_shape = [original_weights['linears.0.bias'].shape[0]] + [original_weights['linears.' + str(i) + '.bias'].shape[0] for i in range(int(len(original_weights)//2))]
weights = [original_weights['linears.' + str(i) + '.weight'].numpy() for i in range(int(len(original_weights)//2))]
biases = [original_weights['linears.' + str(i) + '.bias'].numpy() for i in range(int(len(original_weights)//2))]

neuron_successfully_removed = True
while neuron_successfully_removed:
    neuron_successfully_removed = try_to_combine_neurons(weights, biases)
    if neuron_successfully_removed:
        layer, neuron_1, neuron_2, norm = neuron_successfully_removed
        print('Neurons ' + str(neuron_1) + ' and ' + str(neuron_2) + ' of layer ' + str(layer) + ' have been combined. Norm: ' + str(norm))
    else:
        print('No removable duplicate neuron found.')

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

import torch
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Input the filename of an MLP model to measure the simplicity of.')
parser.add_argument('fname', type=str, help='The name of the model file')
args = parser.parse_args()
fname = args.fname


# Load the weights and biases from fname
original_weights = torch.load(fname, map_location=torch.device('cpu'))
prefix = 'linears.'
original_shape = [original_weights[prefix + '0.bias'].shape[0]] + [original_weights[prefix + str(i) + '.bias'].shape[0] for i in range(int(len(original_weights)//2))]
weights = [original_weights[prefix + str(i) + '.weight'].numpy() for i in range(int(len(original_weights)//2))]
biases = [original_weights[prefix + str(i) + '.bias'].numpy() for i in range(int(len(original_weights)//2))]

norm = sum([np.sum(weight**2) for weight in weights]) + sum([np.sum(bias**2) for bias in biases])

print("Norm: " + str(float(norm)))

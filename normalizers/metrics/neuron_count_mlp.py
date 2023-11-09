import torch
torch.manual_seed(0)
import torch.nn as nn
import numpy as np
import argparse

from neural_verification import MLP, MLPConfig


parser = argparse.ArgumentParser(description='Input the filename of an MLP model to measure the simplicity of.')
parser.add_argument('fname', type=str, help='The name of the model file')
args = parser.parse_args()
fname = args.fname


# Load the weights and biases from fname
original_weights = torch.load(fname, map_location=torch.device('cpu'))
prefix = 'linears.'
original_shape = [original_weights[prefix + '0.weight'].shape[1]] + [original_weights[prefix + str(i) + '.bias'].shape[0] for i in range(int(len(original_weights)//2))]

print("Neurons: " + str(sum(original_shape)))

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


N = 10000 # number of samples
n = 2 # number of numbers per sample

X = torch.normal(0,1,size=(N, n))

y = []
for i in range(N):
    y.append([X[i][0]*X[i][1]])

y = torch.tensor(y)

loss_fn = nn.MSELoss()


# Load the weights and biases from fname
original_weights = torch.load(fname, map_location=torch.device('cpu'))
prefix = 'linears.'
original_shape = [original_weights[prefix + '0.bias'].shape[0]] + [original_weights[prefix + str(i) + '.bias'].shape[0] for i in range(int(len(original_weights)//2))]
weights = [original_weights[prefix + str(i) + '.weight'].numpy() for i in range(int(len(original_weights)//2))]
biases = [original_weights[prefix + str(i) + '.bias'].numpy() for i in range(int(len(original_weights)//2))]

depth = len(weights)
width = weights[0].shape[0]
in_dim = weights[0].shape[1]
out_dim = weights[-1].shape[0]

model = MLP(in_dim=in_dim, out_dim=out_dim, width=width, depth=depth)
model.load_state_dict(original_weights)
    
y_pred = model(X)
epochloss = loss_fn(y_pred, y)
loss = epochloss.item()

print("Loss: " + str(float(loss)))

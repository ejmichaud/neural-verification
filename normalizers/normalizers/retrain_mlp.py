import torch
torch.manual_seed(0)
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import lr_scheduler
import numpy as np
import argparse

from neural_verification import MLP, MLPConfig


parser = argparse.ArgumentParser(description='Input the filename of an MLP model, retrain this model.')
parser.add_argument('fname', type=str, help='The name of the model file')
parser.add_argument('modified_fname', type=str, help='The name of the new model file to be made')
parser.add_argument('-n', '--n_epochs', type=int, help='Number of epochs of SGD to retrain', default=10000)
parser.add_argument('-l', '--learning_rate', type=float, help='Learning rate for retraining', default=0.001)
args = parser.parse_args()
fname = args.fname
modified_fname = args.modified_fname
n_epochs = args.n_epochs
learning_rate = args.learning_rate


N = 10000 # number of samples
n = 2 # number of numbers per sample

X = torch.normal(0,1,size=(N, n))

y = []
for i in range(N):
    y.append([X[i][0]*X[i][1]])

y = torch.tensor(y)

dataset = TensorDataset(X, y)

batch_size = 128
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

loss_fn = nn.MSELoss()


# Load the weights and biases from fname
original_weights = torch.load(fname, map_location=torch.device('cpu'))
prefix = 'linears.'
original_shape = [original_weights[prefix + '0.weight'].shape[1]] + [original_weights[prefix + str(i) + '.bias'].shape[0] for i in range(int(len(original_weights)//2))]
weights = [original_weights[prefix + str(i) + '.weight'].numpy() for i in range(int(len(original_weights)//2))]
biases = [original_weights[prefix + str(i) + '.bias'].numpy() for i in range(int(len(original_weights)//2))]

depth = len(weights)
width = weights[0].shape[0]
in_dim = weights[0].shape[1]
out_dim = weights[-1].shape[0]

model = MLP(in_dim=in_dim, out_dim=out_dim, width=width, depth=depth)
model.load_state_dict(original_weights)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.8)
training_loss = []
for epoch in range(n_epochs):
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()

    scheduler.step()
    
    y_pred = model(X)
    epochloss = loss_fn(y_pred, y)
    training_loss.append(epochloss.item())
#plt.plot(training_loss, label=f'train_loss (d = {width})')
#plt.yscale("log")
#plt.legend(loc='lower left')
#plt.savefig(fname[:-3] + '_retrained.pdf')
torch.save(model.state_dict(), modified_fname)

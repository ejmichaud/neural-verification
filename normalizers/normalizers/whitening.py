import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import yaml
import os
import itertools

from neural_verification import MLP, MLPConfig
from neural_verification import (
    GeneralRNNConfig,
    GeneralRNN,
    cycle,
    FastTensorDataLoader
)

def prepare_xy(x, y, vectorize_input, input_dim, dtype, loss_fn):  # Copied from scripts/rnn_train.py
    # HANDLE X CONVERSIONS
    if vectorize_input:
        x = x.to(torch.int64)
        x = F.one_hot(x, num_classes=input_dim)
    if len(x.shape) == 2: # (batch_size, seq_len)
        x = x.unsqueeze(2) # (batch_size, seq_len, 1)
    x = x.to(dtype)
    # HANDLE Y CONVERSIONS
    if loss_fn == "cross_entropy":
        y = y.to(torch.int64)
    else:
        if len(y.shape) == 2: # (batch_size, seq_len)
            y = y.unsqueeze(2) # (batch_size, seq_len, 1)
        y = y.to(dtype)
    return x, y

parser = argparse.ArgumentParser(description='Input the filename of an RNN, simplify this model.')
parser.add_argument('task', type=str, help='The name of the task')
parser.add_argument('fname', type=str, help='The name of the model file')
parser.add_argument('modified_fname', type=str, help='The name of the new model file to be made')
parser.add_argument('-e', '--epsilon', type=float, help='Error tolerance', default=0.1)
args = parser.parse_args()
task = args.task
fname = args.fname
modified_fname = args.modified_fname
epsilon = args.epsilon


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

class GeneralRNNWithHiddenSaving(GeneralRNN):
    def forward_sequence(self, x):
        """This function takes in a sequence of inputs and returns a sequence of outputs
        as well as the final hidden state."""
        # x shape: (batch_size, sequence_length, input_dim)
        batch_size = x.size(0)
        seq_length = x.size(1)
        hidden = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        assert x.size(2) == self.config.input_dim
        assert x.device == self.device
        hiddens = []
        outs = []
        for i in range(seq_length):
            out, hidden = self.forward(x[:,i,:], hidden)
            hiddens.append(hidden)
            outs.append(out)
        # out shape: (batch_size, sequence_length, output_dim), (batch_size, sequence_length, output_dim), (batch_size, output_dim)
        return torch.stack(outs).permute(1,0,2), torch.stack(hiddens).permute(1,0,2), hiddens[-1]

# Get the RNN weight
rnn = GeneralRNNWithHiddenSaving(config, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
#print(torch.linalg.svd(weights['hmlp.mlp.0.weight'][:,:5]))
#print(torch.matmul(weights['hmlp.mlp.0.weight'][:,:5], torch.matmul(weights['hmlp.mlp.0.weight'][:,:5], torch.matmul(weights['hmlp.mlp.0.weight'][:,:5], torch.matmul(weights['hmlp.mlp.0.weight'][:,:5], torch.matmul(weights['hmlp.mlp.0.weight'][:,:5], weights['hmlp.mlp.0.weight'][:,:5]))))))
#powers_of_W = [torch.eye(5)]
#for i in range(10):
#    print(torch.matmul(torch.matmul(weights['ymlp.mlp.0.weight'], powers_of_W[-1]), weights['hmlp.mlp.0.weight'][:,-1]))
#    powers_of_W.append(torch.matmul(weights['hmlp.mlp.0.weight'][:,:5], powers_of_W[-1]))
#print(weights)
rnn.load_state_dict(weights)
rnn.eval()

# Get the dataset
if not os.path.exists(os.path.join("../tasks/", task, 'data.pt')):
    # run the create_dataset.py script
    os.system(f'python {os.path.join("../tasks/", task, "create_dataset.py")}')

# data is loaded as x_train, y_train, x_test, y_test
dataset = tuple(torch.load(os.path.join("../tasks/", task, 'data.pt')))

# Count up the loss and accuracy
batch_size = 2048
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sequences = [seqs.to(device) for seqs in dataset]
sequences_x_train, \
sequences_y_train, \
sequences_x_test, \
sequences_y_test = sequences

train_loader = FastTensorDataLoader(sequences_x_train, sequences_y_train, batch_size=batch_size, shuffle=False)

loss_fn_name = loss_fn

all_hidden_states = []
rnn.eval()
with torch.no_grad():
    for x, y_target in train_loader:
        x, y_target = prepare_xy(x, y_target, vectorize_input, input_dim, torch.float32, loss_fn_name)
        _, hidden_states, _ = rnn.forward_sequence(x)
        all_hidden_states.append(hidden_states)

all_hidden_states = torch.reshape(torch.cat(all_hidden_states, dim=0), (-1, all_hidden_states[0].shape[-1]))
covariance = torch.matmul(torch.transpose(all_hidden_states, 0, 1), all_hidden_states) / all_hidden_states.shape[0]

S, U = torch.linalg.eigh(covariance)

torch.diag(torch.where(S > epsilon, torch.sqrt(S), torch.tensor(1).type(torch.float32)))
torch.transpose(U, 0, 1)
modified_S_inv = torch.matmul(torch.matmul(U, torch.diag(torch.where(S > epsilon, torch.sqrt(S), torch.tensor(1).type(torch.float32)))), torch.transpose(U, 0, 1))
modified_S = torch.matmul(torch.matmul(U, torch.diag(torch.where(S > epsilon, 1/torch.sqrt(S), torch.tensor(1).type(torch.float32)))), torch.transpose(U, 0, 1))

with torch.no_grad():
    rnn.hmlp.mlp[-1].weight = nn.Parameter(torch.matmul(modified_S, rnn.hmlp.mlp[-1].weight))
    rnn.hmlp.mlp[-1].bias = nn.Parameter(torch.matmul(modified_S, rnn.hmlp.mlp[-1].bias))
    rnn.hmlp.mlp[0].weight = nn.Parameter(torch.cat([torch.matmul(rnn.hmlp.mlp[0].weight[:,:hidden_dim], modified_S_inv), rnn.hmlp.mlp[0].weight[:,hidden_dim:]], dim=1))
    rnn.ymlp.mlp[0].weight = nn.Parameter(torch.matmul(rnn.ymlp.mlp[0].weight, modified_S_inv))

# Put the new weights and biases into modified_model.pt
torch.save(rnn.state_dict(), modified_fname)

print('Transformed RNN hidden space using matrix: ' + str(modified_S))

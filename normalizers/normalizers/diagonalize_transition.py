import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import yaml
import os

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
args = parser.parse_args()
task = args.task
fname = args.fname
modified_fname = args.modified_fname


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
rnn.load_state_dict(weights)
rnn.eval()

# Get the dataset
if not os.path.exists(os.path.join("../tasks/", task, 'data.pt')):
    # run the create_dataset.py script
    os.system(f'python {os.path.join("../tasks/", task, "create_dataset.py")}')

# data is loaded as x_train, y_train, x_test, y_test
dataset = tuple(torch.load(os.path.join("../tasks/", task, 'data.pt')))

# Count up the loss and accuracy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cumulative_product = torch.eye(hidden_dim+input_dim)
for i in range((len(rnn.hmlp.mlp)+1)//2):
    cumulative_product = torch.matmul(rnn.hmlp.mlp[2*i].weight, cumulative_product)
transform = torch.linalg.eig(cumulative_product[:,:hidden_dim])[1].real

with torch.no_grad():
    rnn.hmlp.mlp[-1].weight = nn.Parameter(torch.matmul(torch.linalg.inv(transform), rnn.hmlp.mlp[-1].weight))
    rnn.hmlp.mlp[-1].bias = nn.Parameter(torch.matmul(torch.linalg.inv(transform), rnn.hmlp.mlp[-1].bias))
    rnn.hmlp.mlp[0].weight = nn.Parameter(torch.cat([torch.matmul(rnn.hmlp.mlp[0].weight[:,:hidden_dim], transform), rnn.hmlp.mlp[0].weight[:,hidden_dim:]], dim=1))
    rnn.ymlp.mlp[0].weight = nn.Parameter(torch.matmul(rnn.ymlp.mlp[0].weight, transform))

# Put the new weights and biases into modified_model.pt
torch.save(rnn.state_dict(), modified_fname)

print('Transformed RNN hidden space using matrix: ' + str(transform))


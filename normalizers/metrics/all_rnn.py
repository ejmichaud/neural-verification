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

parser = argparse.ArgumentParser(description='Input the filename of an MLP model to measure the simplicity of.')
parser.add_argument('task', type=str, help='The name of the task')
parser.add_argument('fname', type=str, help='The name of the model file')
args = parser.parse_args()
fname = args.fname
task = args.task

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
    hidden_mlp_width = hidden_dim
if output_mlp_depth >= 2:
    output_mlp_width = weights['ymlp.mlp.0.weight'].shape[0]
else:
    output_mlp_width = output_dim
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

# Count up the weights and neurons
epsilon_threshold = 0.01
with torch.no_grad():
    hmlp_parameter_count = 0
    hmlp_weight_count = 0
    hmlp_bias_count = 0
    hmlp_neuron_count = 0
    hmlp_integer_weight_count = 0
    hmlp_integer_bias_count = 0
    ymlp_parameter_count = 0
    ymlp_weight_count = 0
    ymlp_bias_count = 0
    ymlp_neuron_count = 0
    ymlp_integer_weight_count = 0
    ymlp_integer_bias_count = 0
    weight_norm = 0
    for i, (layer1, layer2) in enumerate(zip(rnn.hmlp.mlp[:-2:2], rnn.hmlp.mlp[2::2])):
        hmlp_neuron_count += np.sum(np.logical_and(np.any(np.abs(layer1.weight.numpy()) > epsilon_threshold, axis=1), np.any(np.abs(layer2.weight.numpy()) > epsilon_threshold, axis=0)).astype(np.int32))
    for i, (layer1, layer2) in enumerate(zip(rnn.ymlp.mlp[:-2:2], rnn.ymlp.mlp[2::2])):
        ymlp_neuron_count += np.sum(np.logical_and(np.any(np.abs(layer1.weight.numpy()) > epsilon_threshold, axis=1), np.any(np.abs(layer2.weight.numpy()) > epsilon_threshold, axis=0)).astype(np.int32))
    for i, layer in enumerate(rnn.hmlp.mlp[::2]):
        hmlp_weight_count += np.sum((np.abs(layer.weight.numpy()) > epsilon_threshold).astype(np.int32))
        hmlp_bias_count += np.sum((np.abs(layer.bias.numpy()) > epsilon_threshold).astype(np.int32))
        hmlp_integer_weight_count += np.sum((np.abs(np.round(layer.weight.numpy()) - layer.weight.numpy()) < epsilon_threshold).astype(np.int32))
        hmlp_integer_bias_count += np.sum((np.abs(np.round(layer.bias.numpy()) - layer.bias.numpy()) < epsilon_threshold).astype(np.int32))
        hmlp_parameter_count += np.sum(np.ones_like(layer.weight.numpy())) + np.sum(np.ones_like(layer.bias.numpy()))
        weight_norm += np.sum(layer.weight.numpy()**2)
    for i, layer in enumerate(rnn.ymlp.mlp[::2]):
        ymlp_weight_count += np.sum((np.abs(layer.weight.numpy()) > epsilon_threshold).astype(np.int32))
        ymlp_bias_count += np.sum((np.abs(layer.bias.numpy()) > epsilon_threshold).astype(np.int32))
        ymlp_integer_weight_count += np.sum((np.abs(np.round(layer.weight.numpy()) - layer.weight.numpy()) < epsilon_threshold).astype(np.int32))
        ymlp_integer_bias_count += np.sum((np.abs(np.round(layer.bias.numpy()) - layer.bias.numpy()) < epsilon_threshold).astype(np.int32))
        ymlp_parameter_count += np.sum(np.ones_like(layer.weight.numpy())) + np.sum(np.ones_like(layer.bias.numpy()))
        weight_norm += np.sum(layer.weight.numpy()**2)
    weight_count = hmlp_weight_count + ymlp_weight_count
    bias_count = hmlp_bias_count + ymlp_bias_count
    integer_weight_count = hmlp_integer_weight_count + ymlp_integer_weight_count
    integer_bias_count = hmlp_integer_bias_count + ymlp_integer_bias_count
    parameter_count = hmlp_parameter_count + ymlp_parameter_count
    neuron_count = hmlp_neuron_count + ymlp_neuron_count
    weight_norm = np.sqrt(weight_norm)

    hidden_writes = np.any(np.abs(rnn.hmlp.mlp[-1].weight.numpy()) > epsilon_threshold, axis=1)
    hidden_hmlp_reads = np.any(np.abs(rnn.hmlp.mlp[0].weight.numpy()[:,:hidden_dim]) > epsilon_threshold, axis=0)
    hidden_ymlp_reads = np.any(np.abs(rnn.ymlp.mlp[0].weight.numpy()) > epsilon_threshold, axis=0)
    hidden_dim = np.sum(np.logical_and(hidden_writes, np.logical_or(hidden_hmlp_reads, hidden_ymlp_reads)))

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
if loss_fn_name == "cross_entropy":
    loss_fn = lambda y_pred, y_target: F.cross_entropy(y_pred, y_target, reduction="none")
elif loss_fn_name == "mse":
    loss_fn = lambda y_pred, y_target: 0.5 * torch.pow(y_pred - y_target, 2)
elif loss_fn_name == "log":
    loss_fn = lambda y_pred, y_target: 0.5 * torch.log(1 + torch.pow(y_pred - y_target, 2))

train_losses = []
train_accuracies = []

sum_loss = 0
sum_accuracy = 0
sum_batches = 0
sum_activated = 0
rnn.eval()
with torch.no_grad():
    for x, y_target in train_loader:
        x, y_target = prepare_xy(x, y_target, vectorize_input, input_dim, torch.float32, loss_fn_name)
        sum_batches += float(x.shape[0]) / batch_size
        y_pred, hidden_states, _ = rnn.forward_sequence(x)
        sum_activated = sum_activated + float(torch.mean((torch.abs(hidden_states) < epsilon_threshold).float()).item()) * x.shape[0] / batch_size
        if loss_fn_name == "cross_entropy":
            y_pred = y_pred.transpose(1, 2) # cross entropy expects (batch_size, num_classes, seq_len)
        batch_losses = loss_fn(y_pred, y_target)
        loss = torch.mean(batch_losses)
        sum_loss = sum_loss + float(loss.item()) * x.shape[0] / batch_size
        if loss_fn_name == "cross_entropy":
            accuracy = (torch.argmax(y_pred, dim=1) == y_target).float().mean()
        else:
            accuracy = (torch.round(y_pred) == y_target).float().mean()
        sum_accuracy = sum_accuracy + float(accuracy.item()) * x.shape[0] / batch_size
        if sum_batches > 10:
            break
loss = sum_loss / sum_batches
accuracy = sum_accuracy / sum_batches
activated_proportion = sum_activated / sum_batches

print("In dim: " + str(input_dim))
print("Out dim: " + str(output_dim))
print("h width: " + str(hidden_mlp_width))
print("h depth: " + str(hidden_mlp_depth))
print("y width: " + str(output_mlp_width))
print("y depth: " + str(output_mlp_depth))
print("Neurons: " + str(neuron_count))
print("Weights: " + str(weight_count))
print("Biases: " + str(bias_count))
print("Integer weights: " + str(integer_weight_count))
print("Integer biases: " + str(integer_bias_count))
print("Parameters: " + str(parameter_count))
print("Weight norm: " + str(weight_norm))
print("Hidden dim: " + str(hidden_dim))
print("Loss: " + str(loss))
print("Accuracy: " + str(accuracy))
print("Activation sparsity: " + str(activated_proportion))
print("")

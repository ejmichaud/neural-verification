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
parser.add_argument('-e', '--epsilon', type=float, help='Error tolerance', default=0.3)
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

# Get the RNN weight
rnn = GeneralRNN(config, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
rnn.load_state_dict(weights)
rnn.eval()

cumulative_product = torch.eye(hidden_dim+input_dim)
for i in range((len(rnn.hmlp.mlp)+1)//2):
    cumulative_product = torch.matmul(rnn.hmlp.mlp[2*i].weight, cumulative_product)
with torch.no_grad():
    W = cumulative_product.numpy()[:,:hidden_dim]

def jcf(W):

    hidden_dim = W.shape[0]
    I = np.eye(hidden_dim)

    D, M = np.linalg.eig(W)  # M D M^{-1}    ### Note: np.linalg.eig(W) doesn't always sort the eigenvalues.

    equivalences = np.abs(D-D[:,np.newaxis]) < epsilon
    for i in range(hidden_dim):
        for j in range(hidden_dim):
            for k in range(hidden_dim):
                assert not (equivalences[i][j] and equivalences[j][k]) or equivalences[i][k]
    eigvals = []
    groups = []
    subspace_dims = []

    subspaces = []
    for i in range(hidden_dim):
        if np.any(equivalences[i,:i]):
            continue
        eigvals.append(np.mean(D[equivalences[i]]))
        groups.append(equivalences[i])
        subspace_dims.append(np.sum(equivalences[i].astype(np.int32)))
        U, S, Vh = np.linalg.svd(W-eigvals[-1]*I)
        rank = np.sum((S > epsilon).astype(np.int32))
        subspaces.append(Vh[rank:,:].T)

    cumulative_kernels = []
    corrected_Ws = []
    for i, (eigval, group, subspace_dim, subspace) in enumerate(zip(eigvals, groups, subspace_dims, subspaces)):
        # construct a projector such that ker(projector) = ker(corrected_W), corrected_W = W - lambda*I
        U, S, Vh = np.linalg.svd(W - eigval*I)
        corrected_W = U[:,:-subspace.shape[1]].dot(np.diag(S[:-subspace.shape[1]])).dot(Vh[:-subspace.shape[1],:])  # = W - lambda*I (with perfect nullspace)
        corrected_Ws.append(corrected_W)

    differential_kernels = []
    for subspace, corrected_W in zip(subspaces, corrected_Ws):
        differential_kernel = subspace
        differential_kernels.append([differential_kernel])
        for i in range(hidden_dim-1):
            U, S, Vh = np.linalg.svd((I-differential_kernel.dot(differential_kernel.T)).dot(corrected_W))
            rank = np.sum((S > epsilon).astype(np.int32))
            differential_kernel_union_subspace = Vh[rank:,:].T
            U, S, Vh = np.linalg.svd(differential_kernel_union_subspace.dot(differential_kernel_union_subspace.T) - subspace.dot(subspace.T))
            differential_kernel = U[:,:hidden_dim-rank - subspace.shape[1]]
            differential_kernels[-1].append(differential_kernel)

        differential_kernels[-1].append(np.zeros([hidden_dim, 0]))

    blocks = []
    for differential_kernel, corrected_W in zip(differential_kernels, corrected_Ws):
        blocks.append([])
        for power, (span1, span2) in reversed(list(enumerate(zip(differential_kernel[:-1], differential_kernel[1:])))):
            assert span1.shape[1] >= span2.shape[1]
            mapped_span2 = np.linalg.svd(corrected_W.dot(span2))[0][:,:span2.shape[1]]
            extra_vectors = [np.linalg.svd(span1.dot(span1.T) - mapped_span2.dot(mapped_span2.T))[2][:span1.shape[1]-span2.shape[1],:].T]
            for power2, kernel in reversed(list(enumerate(differential_kernel[:power]))):
                extra_vectors.append(kernel.dot(kernel.T).dot(corrected_W).dot(extra_vectors[-1]))
            blocks[-1].append(np.stack(extra_vectors[::-1], axis=2).reshape((hidden_dim, (span1.shape[1] - span2.shape[1])*(power+1))))

    transform = np.concatenate([np.concatenate(block, axis=1) for block in blocks], axis=1)

    assert transform.shape[0] == transform.shape[1], transform.shape

    return transform

def accurate_jcf(W):
    net_transform = np.eye(W.shape[0])
    for i in range(3):
        try:
            transform = jcf(np.linalg.inv(net_transform).dot(W).dot(net_transform)).real
            net_transform = net_transform.dot(transform)
        except:
            pass
    return net_transform

transform = accurate_jcf(W)
inv_transform = torch.tensor(np.linalg.inv(transform)).type(rnn.hmlp.mlp[-1].weight.dtype)
transform = torch.tensor(transform).type(rnn.hmlp.mlp[-1].weight.dtype)

with torch.no_grad():
    rnn.hmlp.mlp[-1].weight = nn.Parameter(torch.matmul(inv_transform, rnn.hmlp.mlp[-1].weight))
    rnn.hmlp.mlp[-1].bias = nn.Parameter(torch.matmul(inv_transform, rnn.hmlp.mlp[-1].bias))
    rnn.hmlp.mlp[0].weight = nn.Parameter(torch.cat([torch.matmul(rnn.hmlp.mlp[0].weight[:,:hidden_dim], transform), rnn.hmlp.mlp[0].weight[:,hidden_dim:]], dim=1))
    rnn.ymlp.mlp[0].weight = nn.Parameter(torch.matmul(rnn.ymlp.mlp[0].weight, transform))

# Put the new weights and biases into modified_model.pt
torch.save(rnn.state_dict(), modified_fname)

print('Transformed RNN hidden space using matrix: ' + str(transform))


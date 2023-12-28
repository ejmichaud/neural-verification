
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of layer weight/bias matrices (flattened), and the output
is the activation after applying the layers successively to the zero
vector. The hidden dimension is of size 3, and the activation is tanh.
Sequences are of length 10.

Author: Isaac Liao

-------------------------------------------------------------------
"""

import os
from math import floor, ceil
import random
import gzip

import numpy as np
from tqdm.auto import tqdm
import torch

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    D = int(1e6)
    split = 0.9
    sequences_x = torch.randn((D, 10, 12)) / 2  # dimensions are (batch, sequence length, channels)
    weights = torch.reshape(sequences_x[:,:,:9], (D, 10, 3, 3))
    biases = torch.reshape(sequences_x[:,:,9:], (D, 10, 3, 1))
    sequences_y = []
    activations = torch.zeros((D, 3, 1), dtype=torch.float32)
    for i in range(10):
        activations = torch.tanh(torch.matmul(weights[:,i,:,:], activations) + biases[:,i,:,:])
        sequences_y.append(torch.reshape(activations, (D, 3)))
    sequences_y = torch.stack(sequences_y, dim=1)  # dimensions are (batch, sequence length, channels)
    print(sequences_x[:3,:,:], sequences_y[:3,:,:])

    sequences_x_train = sequences_x[:int(D * split)]
    sequences_x_test = sequences_x[int(D * split):]
    sequences_y_train = sequences_y[:int(D * split)]
    sequences_y_test = sequences_y[int(D * split):]
    # import code; code.interact(local=locals())
    torch.save((
        sequences_x_train, 
        sequences_y_train,
        sequences_x_test, 
        sequences_y_test
    ), os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.pt"))


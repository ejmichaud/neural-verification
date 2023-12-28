
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of unit spin operators defined by corresponding unit
vectors, and the output is the vector representation of the |0>
spinor rotated by all of those spin operators in sequence. Sequences
are of length 10.

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
    unit_vectors = torch.randn((D, 10, 3))
    unit_vectors = unit_vectors / torch.sqrt(torch.sum(unit_vectors**2, dim=2))[:,:,None].type(torch.cfloat)
    pauli_matrices = torch.tensor([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    spin_operators = torch.tensordot(unit_vectors, pauli_matrices, dims=1)
    sequences_x = unit_vectors.type(torch.float)  # dimensions are (batch, sequence length, channels)
    sequences_y = []
    spinors = torch.stack([torch.ones((D,)), torch.zeros((D,))], axis=1).type(torch.cfloat)
    for i in range(10):
        spinors = torch.matmul(spin_operators[:,i,:,:], spinors[:,:,None])[:,:,0]
        sequences_y.append(torch.cat([torch.real(spinors), torch.imag(spinors)], dim=1))
    sequences_y = torch.stack(sequences_y, dim=1)  # dimensions are (batch, sequence length, channels)
#    print(sequences_x[:3,:,:], sequences_y[:3,:,:])

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


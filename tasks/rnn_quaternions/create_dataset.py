
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of unit quaternions, and the output is the unit i hat vector
rotated by all of these quaternion rotations via conjugation, in
sequence. Sequences are of length 10.

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
    versors = torch.randn((D, 10, 4))
    versors = versors / torch.sqrt(torch.sum(versors**2, dim=2))[:,:,None]
    inverse_versors = torch.tensordot(versors, torch.diag(torch.tensor([1,-1,-1,-1], dtype=torch.float32)), dims=1)
    products = torch.tensor([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                             [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]],
                             [[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]],
                             [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]]]).type(torch.float32)
    sequences_x = versors  # dimensions are (batch, sequence length, channels)
    sequences_y = []
    quaternions = torch.cat([torch.zeros((D,1)), torch.ones((D,1)), torch.zeros((D,2))], axis=1).type(torch.float32)
    for i in range(10):
        quaternions = torch.einsum('di,dj,ijk->dk', torch.einsum('di,dj,ijk->dk', versors[:,i,:], quaternions, products), inverse_versors[:,i,:], products)
        sequences_y.append(quaternions[:,1:])
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


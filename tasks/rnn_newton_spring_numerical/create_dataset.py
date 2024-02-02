
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
one dimensional sequence of integer forces, and the output
is a one dimenensional sequence of positions. A spring with constant
k=1 is applied, and the mass is 1. Sequences are of length 10.

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
    split = 0.8
    sequences_x = torch.randint(-10, 11, (D, 10, 1)).type(torch.int32)
    position = torch.zeros((D, 1, 1))
    velocity = torch.zeros((D, 1, 1))
    sequences_y = []
    for i in range(10):
        acceleration = sequences_x[:,i:i+1,:] - position
        velocity = velocity + acceleration
        position = position + velocity
        sequences_y.append(position)
    sequences_y = torch.cat(sequences_y, dim=1).type(torch.int32)

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


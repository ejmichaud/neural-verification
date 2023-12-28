
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of pairs of integers (a, b) in [1, 10), and the output is
their continued fraction up to that point:

    y_i = a_i + b_i/(a_{i-1} + b_2{i-1}/(...(a_2 + b_2/(a_1 + b_1))...))

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
    sequences_x = torch.randint(1, 10, (D, 10, 2), dtype=torch.int64)  # dimensions are (batch, sequence length, channels)
    sequences_y = [torch.ones((D,))]
    for i in range(10):
        sequences_y.append(sequences_x[:,i,0] + sequences_x[:,i,1] / sequences_y[-1])
    sequences_x = sequences_x.type(torch.int8)
    sequences_y = torch.stack(sequences_y[1:], dim=1)[:,:,None]  # dimensions are (batch, sequence length, channels)
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


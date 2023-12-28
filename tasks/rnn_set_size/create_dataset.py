
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of add/remove element operations on a set, and output is
the size of that set. Input is of format (a, b) where a is 1 to
add and 0 to remove, and b is some integer in [0, 10) to add or remove.
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
    sequences_x = torch.stack([torch.randint(0, 2, (D, 10), dtype=torch.int64), torch.randint(0, 10, (D, 10), dtype=torch.int64)], dim=2)  # dimensions are (batch, sequence length, channels)
    elements = torch.zeros((D, 10))
    sequences_y = []
    for i in range(10):
        elements = torch.clip(elements + (sequences_x[:,i,0,None]*2-1)*torch.nn.functional.one_hot(sequences_x[:,i,1], num_classes=10), 0, 1)
        sequences_y.append(torch.sum(elements, dim=1))
    sequences_y = torch.stack(sequences_y, dim=1)[:,:,None]  # dimensions are (batch, sequence length, channels)
#    print(sequences_x[:3,:,:], sequences_y[:3,:,:])
    sequences_x = sequences_x.type(torch.int8)
    sequences_y = sequences_y.type(torch.int8)

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


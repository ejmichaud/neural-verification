
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of 0s and 1s and the output at each sequence position
is whether or not the previous three bits and the current bit are
either 0101 or 1010. 
Elements before the first sequence position are considered to be
zero.
Sequences are of length 10.

Author: Eric Michaud

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

    sequences_x = torch.randint(0, 2, (D, 13), dtype=torch.int8) # pad first to simplify computation, then remove first 2 positions
    sequences_x[:, 0] = 0
    sequences_x[:, 1] = 0
    sequences_x[:, 2] = 0
    sequences_y = torch.zeros_like(sequences_x, dtype=torch.int8)
    for i in range(10):
        # match 0101 or 1010
        iy = i
        ix = i + 3
        sequences_y[:, iy] = (sequences_x[:, ix-3] == 0) & (sequences_x[:, ix-2] == 1) & (sequences_x[:, ix-1] == 0) & (sequences_x[:, ix] == 1) \
                            | (sequences_x[:, ix-3] == 1) & (sequences_x[:, ix-2] == 0) & (sequences_x[:, ix-1] == 1) & (sequences_x[:, ix] == 0)
    
    sequences_x = sequences_x[:, 3:]
    sequences_y = sequences_y[:, :-3]
    # import code; code.interact(local=locals())

    sequences_x_train = sequences_x[:int(D * split)]
    sequences_x_test = sequences_x[int(D * split):]
    sequences_y_train = sequences_y[:int(D * split)]
    sequences_y_test = sequences_y[int(D * split):]

    torch.save((
        sequences_x_train, 
        sequences_y_train,
        sequences_x_test, 
        sequences_y_test
    ), os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.pt"))

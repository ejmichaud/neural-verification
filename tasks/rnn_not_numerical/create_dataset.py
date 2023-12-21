
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of 0s and 1s and the output at each sequence position
is the NOT of the current sequence element. 
Sequences are lists of integers {0, 1} of length 10.

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

    sequences_x = torch.randint(0, 2, (D, 10), dtype=torch.int8)
    sequences_y = torch.zeros_like(sequences_x, dtype=torch.int8)
    for i in range(10):
        sequences_y[:, i] = (sequences_x[:, i] + 1) % 2
    
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


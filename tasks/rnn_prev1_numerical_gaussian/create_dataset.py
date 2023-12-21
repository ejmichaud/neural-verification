
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of integers and the output at each sequence position
is the minimum value of the current and previous sequence
element. Sequences are lists of integers in [0, 100)
of length 10.

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

    # sample from a standard normal
    sequences_x = torch.randn((D, 10), dtype=torch.float32)
    # sequences_x = torch.randint(0, 100, (D, 10), dtype=torch.int8)
    sequences_y = torch.zeros((D, 10), dtype=torch.float32)
    for i in tqdm(range(10)):
        sequences_y[:, i] = sequences_x[:, max(0, i-1)]
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

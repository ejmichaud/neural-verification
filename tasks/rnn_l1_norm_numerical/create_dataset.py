
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of integers and the output at each sequence position
is the value of the sequence position after being l1 normalized.
Sequences are lists of integers in [1, 100) of length 10.

-------------------------------------------------------------------
"""

from math import floor, ceil
import random
import gzip

import numpy as np
from tqdm.auto import tqdm
import torch

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    D = int(1e6)
    split = 0.9
    powers_of_2 = [2 ** i for i in range(7)]
    sequences_x = torch.randint(1, 100, (D, 10), dtype=torch.int8)
    cumsum_x = torch.cumsum(sequences_x, dim=1)[0]
    sequences_y = sequences_x / cumsum_x
    print(sequences_x[:5], sequences_y[:5])
    sequences_x_train = sequences_x[:int(D * split)]
    sequences_x_test = sequences_x[int(D * split):]
    sequences_y_train = sequences_y[:int(D * split)]
    sequences_y_test = sequences_y[int(D * split):]
    torch.save((
        sequences_x_train, 
        sequences_y_train,
        sequences_x_test, 
        sequences_y_test
    ), "data.pt")
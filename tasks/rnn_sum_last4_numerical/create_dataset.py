
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of integers and the output at each sequence position
is the sum of the previosu element and the element in the
current position of the input sequence.
Sequences are lists of integers in [0, 100) of length 10.

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

    sequences_x = torch.randint(0, 100, (D, 10), dtype=torch.int16)
    sequences_y = torch.zeros_like(sequences_x, dtype=torch.int16)
    for i in range(10):
        sequences_y[:, i] = sequences_x[:, max(0, i-3):i+1].sum(dim=-1)

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
    ), "data.pt")

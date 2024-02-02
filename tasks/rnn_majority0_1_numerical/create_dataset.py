
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of integers {0, 1} and the output at each sequence
position is the integer which has occurred the most so far in the
sequence. Ties break towards the lower number. Sequences are 
of length 10.

Author: Eric Michaud

-------------------------------------------------------------------
"""

import os
from math import floor, ceil
from collections import defaultdict
import random
import gzip

import numpy as np
from tqdm.auto import tqdm
import torch

def y(x):
    """Takes in a PyTorch 1d array and outputs the output
    sequence from this input, where the output at 
    each sequence position is the integer which has
    occurred the most so far in the sequence. Ties
    break towards the lower number.
    """
    y = torch.zeros_like(x)
    counts = defaultdict(int)
    for i in range(x.shape[0]):
        counts[x[i].item()] += 1
        y[i] = max(sorted(counts), key=counts.get)
    return y

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    D = int(1e6)
    split = 0.9

    # we can use int8 to save memory since our max value is 100
    # and int8 can store values up to 127
    sequences_x = torch.randint(0, 2, (D, 10), dtype=torch.int8)
    sequences_y = torch.zeros((D, 10), dtype=torch.int8)
    for i in tqdm(range(D)):
        sequences_y[i] = y(sequences_x[i])

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


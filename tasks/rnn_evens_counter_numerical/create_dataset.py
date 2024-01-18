
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of integers in {0, 1, 2, ..., 9} and the output at each
sequence position is the number of even integers seen so far. 
Sequences are of length 10.

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

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    D = int(1e6)
    split = 0.9

    # we can use int8 to save memory since our max value is 100
    # and int8 can store values up to 127
    sequences_x = torch.randint(0, 10, (D, 10), dtype=torch.int8)
    sequences_y = torch.zeros((D, 10), dtype=torch.int8)
    vowels = sequences_x == 0
    vowels += sequences_x == 2
    vowels += sequences_x == 4
    vowels += sequences_x == 6
    vowels += sequences_x == 8
    sequences_y = vowels.cumsum(dim=1)
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

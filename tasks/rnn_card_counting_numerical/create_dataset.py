
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of cards labelled by integers 0-12 inclusive, and each
card repeats exactly 4 times over the sequence of length 52. The
output is the running sum of somebody who is doing basic card-counting.

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
    sequences_x = torch.stack([torch.randperm(52) for i in range(D)], axis=0).type(torch.int64) % 13
    sequences_y = torch.cumsum(torch.take(torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, -1]), sequences_x), axis=1)
    sequences_x = sequences_x[:,:,None].type(torch.int32)
    sequences_y = sequences_y[:,:,None].type(torch.int32)
#    print(sequences_x[:2,:].tolist())
#    print(sequences_y[:2,:].tolist())

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

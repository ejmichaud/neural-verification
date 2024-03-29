
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of integers and the output at each sequence position
is 1 if the current element is the same as the previous element. 
For the first element, the output is zero. 
Sequences are lists of integers in [0, 8)
of length 10.

Author: Carl Guo

Note from Eric Michaud: edited so that the first output is 1
if the first element is 0. This is as if the -1th input
element was 0.

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

    # we can use int8 to save memory since our max value is 100
    # and int8 can store values up to 127
    sequences_x = torch.randint(0, 8, (D, 10), dtype=torch.int8)
    sequences_x = torch.unique(sequences_x, dim=0, sorted=False)
    
    idx = torch.randperm(sequences_x.shape[0])
    sequences_x = sequences_x[idx]
    D = sequences_x.shape[0]
    sequences_y = torch.zeros((D, 10), dtype=torch.int8)
    for i in tqdm(range(1, 10)):
        sequences_y[:, i] = sequences_x[:, i] == sequences_x[:, i - 1]
    sequences_y[:, 0] = 0
    # get indices where the first element is 0
    indices0 = torch.where(sequences_x[:, 0] == 0)[0]
    sequences_y[indices0, 0] = 1
    # print(sequences_x[:5], sequences_y[:5])
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


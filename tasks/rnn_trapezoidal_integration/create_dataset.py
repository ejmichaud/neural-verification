
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of (delta x, y) pairs of a piecewise function, and the output
is the integral of the piecewise function created by those points.
Sequences are of length 10. The first delta x value is meaningless.
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
    sequences_x = torch.stack([torch.abs(torch.randn((D, 10))), torch.randn((D, 10))], dim=2)  # dimensions are (batch, sequence length, channels)
    sequences_y = torch.cat([torch.zeros((D, 1)), torch.cumsum((sequences_x[:,1:,1] + sequences_x[:,:-1,1]) * sequences_x[:,1:,0] / 2, dim=1)], dim=1)[:,:,None]  # dimensions are (batch, sequence length, channels)
#    print(sequences_x[:3,:,:], sequences_y[:3,:,:])

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




"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
one dimensional continuous-brightness image with one color channel,
and the output is a dithered brightness-quantized version. Sequences
are of length 10.

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
    sequences_x = torch.randn((D, 10))
    sequences_y = torch.round(torch.cumsum(sequences_x, dim=1))
    sequences_y = torch.cat([torch.zeros((D, 1)), sequences_y], dim=1)
    sequences_y = sequences_y[:,1:] - sequences_y[:,:-1]
    sequences_y = sequences_y.type(torch.int8)
#    print(sequences_x[0,:])
#    print(sequences_y[0,:])

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


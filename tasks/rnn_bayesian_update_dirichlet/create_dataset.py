
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of observations of a categorical variable and the output
is the mean on a Dirichlet distribution posterior of the categorical
distribution parameters. The prior is the uniform distribution.
There are 3 classes in the categorical distribution. Sequences are
of length 10.

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
    sequences_x = torch.randint(0, 3, (D, 10), dtype=torch.int64)
    sequences_y = 1 + torch.cumsum(torch.nn.functional.one_hot(sequences_x, num_classes=3), dim=1).type(torch.float64)
    sequences_y = sequences_y / torch.sum(sequences_y, dim=2)[:,:,None]  # dimensions are (batch, sequence length, channels)
#    print(sequences_x[:3,:], sequences_y[:3,:])
    sequences_x = sequences_x.type(torch.int8)

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


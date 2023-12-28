
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of random normal values, and the output is the empirical
skewness of all the elements up to that point. Sequences are of
length 10.

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
    sum_1 = torch.cumsum(torch.ones_like(sequences_x), dim=1)
    sum_x = torch.cumsum(sequences_x, dim=1)
    sum_xx = torch.cumsum(sequences_x**2, dim=1)
    sum_xxx = torch.cumsum(sequences_x**3, dim=1)
    sequences_y = (sum_xxx/sum_1-3*sum_xx*sum_x/sum_1**2+2*sum_x**3/sum_1**3)/(sum_xx/sum_1 - (sum_x/sum_1)**2)**(3/2)

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

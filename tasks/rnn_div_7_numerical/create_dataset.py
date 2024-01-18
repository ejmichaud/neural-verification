
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
dividend in base 2 reverse order, and the output is the quotient (div by 7)
in base 2 reverse order. For example, we have 10011010/111=00010110,

  in | out
  --------
   1 |  0
   0 |  0
   0 |  0
   1 |  1
   1 |  0
   0 |  1
   1 |  1
   0 |  0

Sequences are of length 10, instead of 7 like in this example.

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
    sequences_x = torch.randint(0, 2, (D, 10, 1), dtype=torch.int64)
    cumulant = 0
    sequences_y = [torch.zeros((D, 1))]
    for i in range(10):
        cumulant = cumulant * 2 + sequences_x[:,i,:]
        cumulant, remainder = cumulant % 7, cumulant // 7
        sequences_y.append(remainder)
    sequences_y = torch.stack(sequences_y, dim=1)
    sequences_x = sequences_x.type(torch.int32)
    sequences_y = sequences_y.type(torch.int32)

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



"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
divisor followed by a dividend in base 10 reverse order, and the
output is the quotient in base 10 reverse order. For example, we
have 1024/2=512,

  in | out
  --------
   2 |  0
   1 |  0
   0 |  5
   2 |  1
   4 |  2

Sequences are of length 10, instead of 5 like in this example.

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
    divisor = torch.randint(1, 10, (D,), dtype=torch.int64)
    dividend = torch.randint(0, 10, (D, 9), dtype=torch.int64)
    sequences_x = torch.cat([divisor[:,None], dividend], dim=1)
    cumulant = 0
    sequences_y = [torch.zeros((D,))]
    for i in range(1, 10):
        cumulant = cumulant * 10 + sequences_x[:,i]
        cumulant, remainder = cumulant % divisor, cumulant // divisor
        sequences_y.append(remainder)
    sequences_y = torch.stack(sequences_y, dim=1)
#    print(sequences_x[:3,:], sequences_y[:3,:])
    sequences_x = sequences_x.type(torch.int8)
    sequences_y = sequences_y.type(torch.int8)

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


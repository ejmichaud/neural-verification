
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of 0s and 1s and the output at each sequence position
is the parity of the current and previous two inputs. At the first
position, the output is the XOR of the input and zero. At the second
position the output is the parity zero, first element, and second
element. Sequences are of length 10.

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

    sequences_x = torch.randint(0, 2, (D, 10), dtype=torch.int8)
    sequences_y = torch.zeros_like(sequences_x, dtype=torch.int8)
    for i in range(10):
        sequences_y[:, i] = torch.sum(sequences_x[:, max(i-2, 0):i+1], dim=-1) % 2
        # if i == 0:
        #     sequences_y[:, i] = sequences_x[:, i]
        # else:
            # sequences_y[:, i] = sequences_x[:, i] | sequences_x[:, i-1]
            # sequences_y[:, i] = sequences_x[:, i] ^ sequences_x[:, i-1]
            # compute the parity of 0:i

    sequences_x_train = sequences_x[:int(D * split)]
    sequences_x_test = sequences_x[int(D * split):]
    sequences_y_train = sequences_y[:int(D * split)]
    sequences_y_test = sequences_y[int(D * split):]
    import code; code.interact(local=locals())
    torch.save((
        sequences_x_train, 
        sequences_y_train,
        sequences_x_test, 
        sequences_y_test
    ), os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.pt"))

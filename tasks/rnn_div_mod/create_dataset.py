
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of integers in [0, 19), and the output is the cumulative
division mod 19. NOTE: you can't divide by zero, so zero is not
included in this dataset, and all the input and output numbers are
shifted down by one. Sequences are of length 10.

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
    inverses_table = torch.Tensor([x**17 % 19 for x in range(19)]).type(torch.int64)
#    print(inverses_table)
    sequences_x = torch.randint(1, 19, (D, 10), dtype=torch.int64)
    inverted_x = torch.gather(torch.tile(inverses_table[:,None, None], (1, sequences_x.size(0), sequences_x.size(1))), 0, sequences_x[None,:,:])[0,:,:].type(torch.int8)
    sequences_y = torch.cumprod(inverted_x, dim=1) % 19
    sequences_x = (sequences_x - 1).type(torch.int8)
    sequences_y = (sequences_y - 1).type(torch.int8)
#    print(sequences_x[:2,:])
#    print(sequences_y[:2,:])

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

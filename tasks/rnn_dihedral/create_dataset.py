
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of dihedral group elements and the output is the cumulative
group operation of all the dihedral group elements up until that point.
Dihedral group is of order 7, having 14 elements. Sequences are of
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
    cayley_table = torch.Tensor([
            [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13], 
            [ 1,  2,  3,  4,  5,  6,  0, 13,  7,  8,  9, 10, 11, 12], 
            [ 2,  3,  4,  5,  6,  0,  1, 12, 13,  7,  8,  9, 10, 11], 
            [ 3,  4,  5,  6,  0,  1,  2, 11, 12, 13,  7,  8,  9, 10], 
            [ 4,  5,  6,  0,  1,  2,  3, 10, 11, 12, 13,  7,  8,  9], 
            [ 5,  6,  0,  1,  2,  3,  4,  9, 10, 11, 12, 13,  7,  8], 
            [ 6,  0,  1,  2,  3,  4,  5,  8,  9, 10, 11, 12, 13,  7], 
            [ 7,  8,  9, 10, 11, 12, 13,  0,  1,  2,  3,  4,  5,  6], 
            [ 8,  9, 10, 11, 12, 13,  7,  6,  0,  1,  2,  3,  4,  5], 
            [ 9, 10, 11, 12, 13,  7,  8,  5,  6,  0,  1,  2,  3,  4], 
            [10, 11, 12, 13,  7,  8,  9,  4,  5,  6,  0,  1,  2,  3], 
            [11, 12, 13,  7,  8,  9, 10,  3,  4,  5,  6,  0,  1,  2], 
            [12, 13,  7,  8,  9, 10, 11,  2,  3,  4,  5,  6,  0,  1], 
            [13,  7,  8,  9, 10, 11, 12,  1,  2,  3,  4,  5,  6,  0], 
    ]).type(torch.int64)
    sequences_x = torch.randint(0, 14, (D, 10), dtype=torch.int64)
    sequences_y = [sequences_x[:,0]]
    for i in range(1, 10):
        sequences_y.append(torch.gather(torch.index_select(cayley_table, 0, sequences_y[-1]), 1, sequences_x[:,i,None])[:,0])
    sequences_y = torch.stack(sequences_y, dim=1)
    sequences_x = sequences_x.type(torch.int8)
    sequences_y = sequences_y.type(torch.int8)
#    print(sequences_x[:2,:], sequences_y[:2,:])

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


"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of permutation group elements and the output is the cumulative
group operation of all the permutation group elements up until that point.
Permutation group is of order 4, having 24 elements. Sequences are of
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
from itertools import permutations

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    D = int(1e6)
    split = 0.9
    perms = list(sorted(list(set(permutations(tuple(range(4)))))))
#    print(perms)
    cayley_table = torch.Tensor([[perms.index(tuple(perms[i][k] for k in perms[j])) for j in range(24)] for i in range(24)]).type(torch.int64)
#    print(cayley_table)
    sequences_x = torch.randint(0, 24, (D, 10), dtype=torch.int64)
    sequences_y = [sequences_x[:,0]]
    for i in range(1, 10):
        sequences_y.append(torch.gather(torch.index_select(cayley_table, 0, sequences_y[-1]), 1, sequences_x[:,i,None])[:,0])
    sequences_y = torch.stack(sequences_y, dim=1)
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

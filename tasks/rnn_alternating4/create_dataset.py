
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of elements of an alternating group of order 4, and the
output is the cumulative group operation over the sequence. The group
has 12 elements. Sequences are of length 10.

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
    alternating_elements = torch.Tensor([i for i in range(24) if sum([int(perms[i][x]>perms[i][y]) for y in range(4) for x in range(y)]) % 2 == 0]).type(torch.int64)
    nonalternating_elements = torch.Tensor([i for i in range(24) if sum([int(perms[i][x]>perms[i][y]) for y in range(4) for x in range(y)]) % 2 != 0]).type(torch.int64)
    alternating_indices = torch.argsort(torch.cat([alternating_elements, nonalternating_elements], dim=0))
#    print(alternating_elements)
#    print(alternating_indices)
    sequences_x = torch.randint(0, alternating_elements.size(0), (D, 10), dtype=torch.int64)
    sequences_y = [torch.index_select(alternating_elements, 0, sequences_x[:,0])]
    for i in range(1, 10):
        sequences_y.append(torch.gather(torch.index_select(cayley_table, 0, sequences_y[-1]), 1, torch.index_select(alternating_elements, 0, sequences_x[:,i])[:,None])[:,0])
    sequences_x = sequences_x.type(torch.int8)
#    print(sequences_x[:2,:])
    sequences_y = torch.stack(sequences_y, dim=1)
#    print(sequences_y[:2,:])
    sequences_y = torch.gather(torch.tile(alternating_indices[:,None, None], (1, sequences_y.size(0), sequences_y.size(1))), 0, sequences_y[None,:,:])[0,:,:].type(torch.int8)
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

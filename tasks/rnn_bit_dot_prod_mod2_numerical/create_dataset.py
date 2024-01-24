"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
a sequence of 2d vectors with entries either 0 or 1. This represents
two binary sequences. The output at each sequence position
is the dot product of the bit strings up until that sequence
position mod 2.
Sequences are of length 10.

Author: Eric Michaud

-------------------------------------------------------------------
"""

import os
import random
from tqdm.auto import tqdm
import numpy as np
import torch

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    D = int(1e6)
    split = 0.9

    sequences_x = torch.randint(0, 2, (D, 10, 2), dtype=torch.int8)
    sequences_y = torch.zeros(D, 10, dtype=torch.int8)
    for i in tqdm(range(10)):
        sequences_y[:, i] = torch.sum(sequences_x[:, :i+1, 0] * sequences_x[:, :i+1, 1], dim=1) % 2
    # import code; code.interact(local=locals())
    sequences_x_train = sequences_x[:int(D * split)]
    sequences_x_test = sequences_x[int(D * split):]
    sequences_y_train = sequences_y[:int(D * split)]
    sequences_y_test = sequences_y[int(D * split):]
    torch.save((
        sequences_x_train, 
        sequences_y_train,
        sequences_x_test, 
        sequences_y_test
    ), os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.pt"))


"""
----------------------- DATASET DESCRIPTION -----------------------
This dataset is a sequence prediction task where the input is a
sequence of integers and the output at each sequence position
is 0 if it's not a power of 2 and 1 if it is a power of 2.
Sequences are lists of integers in [0, 100) of length 10.
The sequences are 70% random integers and 30% powers of 2.

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

    D = int(1e6)
    split = 0.9
    powers_of_2 = [2 ** i for i in range(7)]
    sequences_x = torch.randint(0, 100, (D, 10), dtype=torch.int16)
    sequences_y = torch.zeros_like(sequences_x)
    for i in tqdm(range(sequences_x.shape[0])):
        random_inserts = torch.rand((10,))
        sequences_x[i, random_inserts > 0.7] = powers_of_2[random.randint(0, 6)]
        for j, power in enumerate(powers_of_2):
            sequences_y[i, sequences_x[i] == power] = 1
    print(sequences_x[:5], sequences_y[:5])

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
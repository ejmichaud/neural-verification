
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of integers and the output at each sequence position
is the previous element in the sequence. For the first element,
the output is zero. Sequences are lists of integers in [0, 100)
of length 10.

-------------------------------------------------------------------
"""

import os
from math import floor, ceil
import random
import gzip

import numpy as np
from tqdm.auto import tqdm
import torch

val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1
        ]
syms = [
        "M", "CM", "D", "CD",
        "C", "XC", "L", "XL",
        "X", "IX", "V", "IV",
        "I"
        ]
    
def int_to_roman(x):
    
    y = ""
    i = 0
    while x > 0:
        for _ in range(x // val[i]):
            y += syms[i]
            x -= val[i]
        i += 1
    return y


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    D = 4000
    split = 0.9

    # we can use int8 to save memory since our max value is 100
    # and int8 can store values up to 127
    sequences_x = list(range(1, D))
    sequences_y = [int_to_roman(x) for x in sequences_x]
    combined = list(zip(sequences_x, sequences_y))
    random.shuffle(combined)
    sequences_x[:], sequences_y[:] = zip(*combined)
    sequences_x = torch.tensor(sequences_x)
    
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

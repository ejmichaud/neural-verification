
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of integers and the output at each sequence position
is the number of unique integers in the context of the current 
element to the left two elements.
For the first element, the output is 1. 
Sequences are lists of integers in [0, 10)
of length 10.

-------------------------------------------------------------------
"""

from math import floor, ceil
import random
import gzip

import numpy as np
from tqdm.auto import tqdm
import torch

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    D = int(5e5)
    split = 0.9

    # we can use int8 to save memory since our max value is 100
    # and int8 can store values up to 127
    max_int = 10
    
    sequences_x = None
    sequences_y = None
    for _ in tqdm(range(D)):
        sequence_x = torch.randperm(max_int)
        for i in range(1, len(sequence_x)):
            random_num = random.random()
            if random_num > 0.5 and random_num < 0.9: 
                sequence_x[i] = sequence_x[i - 1]
            elif random_num > 0.9:
                sequence_x[i] = sequence_x[i - 1]
                if i + 1 < len(sequence_x):
                    sequence_x[i +1] = sequence_x[i - 1]
                    i+=1
        sequence_x_shift1 = torch.roll(sequence_x, 1)
        sequence_x_shift1[0] = 0
        sequence_x_shift2 = torch.roll(sequence_x, 2)
        sequence_x_shift2[:2] = 0
        sequence_y = torch.eq(sequence_x_shift1, sequence_x).int()
        sequence_y += torch.eq(sequence_x_shift2, sequence_x).int()

        if sequences_x is None:
            sequences_x = sequence_x
            sequences_y = sequence_y
        else: 
            sequences_x = torch.vstack((sequences_x, sequence_x))
            sequences_y = torch.vstack((sequences_y, sequence_y))
    print(sequences_x.shape)
    print(len(torch.unique(sequences_x, dim=0)))
    print(sequences_x[:5])
    print(sequences_y[:5])
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
    ), "data.pt")


"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of integers and the output at each sequence position
is the sum of the elements up to including the element at the 
current position of the input sequence. 
Sequences are lists of integers in [0, 100) of length 10.

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

    D = int(1e6)
    split = 0.9

    sequences_x = torch.randint(0, 10, (D, 10), dtype=torch.int16)
    sequences_y = torch.zeros_like(sequences_x)

    for row_idx, tensor in enumerate(tqdm(sequences_x)):
        unique_dict = {}
        count = 0

        for idx, element in enumerate(tensor):
            if element.item() not in unique_dict:
                unique_dict[element.item()] = count
                count += 1
            sequences_y[row_idx, idx] = count

    sequences_y = torch.tensor(sequences_y, dtype=torch.int16)
    print(sequences_x[0], sequences_y[0])

    sequences_x_train = sequences_x[:int(D * split)]
    sequences_x_test = sequences_x[int(D * split):]
    sequences_y_train = sequences_y[:int(D * split)]
    sequences_y_test = sequences_y[int(D * split):]
    print(sequences_x_train[0], sequences_y_train[0])
    torch.save((
        sequences_x_train, 
        sequences_y_train,
        sequences_x_test, 
        sequences_y_test
    ), "data.pt")


"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
bit sequence, and the output is the running value of the Bluetooth
CRC-16 checksum (coming from the command cksum) of the bits up to that
point, as though the bits were part of a file. Sequences are of
length 100.

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

    D = int(1e4)
    split = 0.9
    generator = torch.Tensor(list(map(int, bin(0x11021)[2:])))
    n_generator_digits = generator.size(0)
    sequences_x = torch.randint(0, 2, (D, 100, 1), dtype=torch.int64)  # dimensions are (batch, sequence length, channels)
    remainder = torch.zeros((D, n_generator_digits-1))
    sequences_y = []
    for i in range(1, 100):
        remainder = torch.cat([remainder, sequences_x[:,i:i+1,0]], axis=1)
        remainder = (remainder + remainder[:,:1]*generator)[:,1:] % 2
        sequences_y.append(remainder)
    sequences_x = sequences_x.type(torch.int8)
    sequences_y = torch.stack(sequences_y, dim=1)  # dimensions are (batch, sequence length, channels)
    sequences_y = sequences_y.type(torch.int8)
#    print(sequences_x[:3,:64,0])
#    print("\n".join(["".join(list(map(lambda i: " " if i==0 else "X", x))) for x in sequences_y[0,:64,:].numpy().tolist()]))

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



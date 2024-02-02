
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of bits {0, 1} and the output at each sequence
position is whether the sequence so far is a palindrome. 1 if a
palindrome, else 0.

Author: Eric Michaud

-------------------------------------------------------------------
"""

import os
import random
from math import floor, ceil
from tqdm.auto import tqdm
import numpy as np
import torch


def sample_palindrome(length):
    """
    Sample a palindrome of the given length.
    """
    characters = np.random.choice(list(range(2)), size=length).tolist()
    return characters[:ceil(length / 2)] + characters[:floor(length / 2)][::-1]

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    D = int(1e6)
    split = 0.9

    sequences_x = torch.zeros((D, 10), dtype=torch.int8)
    sequences_y = torch.zeros_like(sequences_x, dtype=torch.int8)

    for i in tqdm(range(D)):
        length = np.random.randint(2, 11)
        palindrome = torch.tensor(sample_palindrome(length), dtype=torch.int8)
        background = torch.randint(0, 2, (10 - length,), dtype=torch.int8)
        seq = torch.cat([palindrome, background])
        sequences_x[i] = seq
        # import code; code.interact(local=locals())
        for j in range(10):
            forward = seq[:j+1]
            # import code; code.interact(local=locals())
            # flip forward
            backward = torch.flip(forward, [0])
            # backward = forward[::-1]
            if torch.all(forward == backward):
                sequences_y[i, j] = 1
            else:
                sequences_y[i, j] = 0
    
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

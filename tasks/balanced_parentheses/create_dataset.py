
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of 0s and 1s, with 0 representing an open parenthesis "("
and 1 representing a closed parenthesis ")". The output at each
sequence position is 1 if the parentheses up to that position are
balanced, and 0 otherwise. Sequences are of length 10.

Author: Anish Mudide and Eric Michaud

-------------------------------------------------------------------
"""

import os
import random
from tqdm.auto import tqdm
import numpy as np
import torch

def check_parentheses_balance(sequence):
    """
    Check if the sequence of parentheses represented by 0s (open) and 1s (close) is balanced.
    Returns a tensor of the same length with 1s at positions where the sequence is balanced and 0s otherwise.
    """
    balance = 0
    balance_sequence = torch.zeros_like(sequence, dtype=torch.int8)
    for i, char in enumerate(sequence):
        if char == 0:  # open parenthesis
            balance += 1
        else:  # close parenthesis
            balance -= 1
        if balance == 0:
            balance_sequence[i] = 1
        if balance < 0:
            break

    return balance_sequence

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    D = int(1e6)
    split = 0.9

    sequences_x = torch.randint(0, 2, (D, 10), dtype=torch.int8)
    sequences_y = torch.zeros_like(sequences_x, dtype=torch.int8)
    for i in tqdm(range(D)):
        sequences_y[i] = check_parentheses_balance(sequences_x[i])

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

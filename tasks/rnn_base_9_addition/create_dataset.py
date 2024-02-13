
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of 2d vectors with entries in base 9. This represents
two sequences in base 9. The output is the sum of the two strings.
The least significant bits come first, on the left of the sequence.
Sequences are of length 10.

Note that the output is always 10 digits long, so if the sum requires
more digits, the most significant digits will be lost.

Author: Vedang Lad

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

    base = 9
    sequences_x = torch.randint(0, base, (D, 10, 2), dtype=torch.int8)
    sequences_y = torch.zeros(D, 10, dtype=torch.int8)
    carries = torch.zeros(D, 11, dtype=torch.int8)
    for i in tqdm(range(10)):
        sequences_y[:, i] = (sequences_x[:, i, 0] + sequences_x[:, i, 1] + carries[:, i]) % base
        carries[:, i+1] = (sequences_x[:, i, 0] + sequences_x[:, i, 1] + carries[:, i]) // base
            
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

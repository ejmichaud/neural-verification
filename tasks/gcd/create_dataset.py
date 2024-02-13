"""
----------------------- DATASET DESCRIPTION -----------------------
This dataset is a Greatest Common Divisor (GCD) task where the input
is a pair of integers and the label is the GCD of the
two integers. All three integers are represented as binary strings.

All integers are between 0 and 2^16 - 1. The length of each string
is 16, with leading zeros if necessary.

To generate the input integers, we sample uniformly from the range
0 to 2^16 - 1. To generate the label, we use the Euclidean algorithm
to compute the GCD of the two integers.
-------------------------------------------------------------------
"""

import os
from math import gcd
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def int_to_binary_str(x: int, n_bits: int):
    if x < 0:
        raise ValueError("x must be non-negative")
    if x >= 2**n_bits:
        raise ValueError("x must be less than 2**n_bits")
    res = format(x, f"0{n_bits}b")
    return res

def create_dataset_entry(n_bits: int):
    """Creates a pair of integers and their GCD"""
    x = np.random.randint(0, 2**16)
    y = np.random.randint(0, 2**16)
    return x, y, gcd(x, y)

if __name__ == "__main__":
    np.random.seed(2348970)

    n_bits = 16
    D = int(1e4)
    split = 0.8

    pairs = []
    gcds = []
    for _ in tqdm(range(D)):
        a, b, c = create_dataset_entry(n_bits)
        pairs.append((int_to_binary_str(a, n_bits), int_to_binary_str(b, n_bits)))
        gcds.append(int_to_binary_str(c, n_bits))

    pairs_train = pairs[:int(split * D)]
    pairs_test = pairs[int(split * D):]
    gcds_train = gcds[:int(split * D)]
    gcds_test = gcds[int(split * D):]


    # save dataset to csv
    df = pd.DataFrame({"first": [pair[0] for pair in pairs_train], "second": [pair[1] for pair in pairs_train], "gcd": gcds_train})
    df.to_csv("gcd_train.csv", index=False)
    df = pd.DataFrame({"first": [pair[0] for pair in pairs_test], "second": [pair[1] for pair in pairs_test], "gcd": gcds_test})
    df.to_csv("gcd_test.csv", index=False)
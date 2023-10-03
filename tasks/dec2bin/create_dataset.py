"""
----------------------- DATASET DESCRIPTION -----------------------
This dataset is a Decimal to Binary (Dec2Bin) task where the input
is a decimal integer and the label is the binary representation
of the integer. The binary representation is a string of 0s and 1s.

All integers are between 0 and 2^16 - 1. The length of each string
is 16, with leading zeros if necessary.

To generate the input integers, we sample uniformly from the range
0 to 2^16 - 1. To generate the label, we use the built-in bin()
function.
-------------------------------------------------------------------
"""

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
    x = np.random.randint(0, 2**16)
    y = int_to_binary_str(x, n_bits)
    print(x, y)
    return x, y

if __name__ == "__main__":
    np.random.seed(2348970)

    n_bits =   16
    D = 5000 # dataset size
    split = 0.8

    entries = []
    for _ in tqdm(range(D)):
        x, y = create_dataset_entry(n_bits)
        entries.append((x, y))

    entries_train = entries[:int(split * D)]
    entries_test = entries[int(split * D):]

    # save dataset to csv
    df = pd.DataFrame({"integer": [entry[0] for entry in entries_train], "binary": [entry[1] for entry in entries_train]})
    df.to_csv("dec2bin_train.csv", index=False)
    df = pd.DataFrame({"integer": [entry[0] for entry in entries_test], "binary": [entry[1] for entry in entries_test]})
    df.to_csv("dec2bin_test.csv", index=False)
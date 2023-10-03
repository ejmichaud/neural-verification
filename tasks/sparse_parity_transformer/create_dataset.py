
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a binary classification task on binary strings of
length 20. The label is the parity of first three bits in the
input string. After the input string, there is a "|" character, 
followed by the answer, which is either "0" or "1". We generate
the input string by sampling each bit uniformly at random.

-------------------------------------------------------------------
"""

from math import floor, ceil
import random

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from neural_verification import Tokenizer
from config import tokenizer_vocabulary

def create_string() -> str:
    """Creates a dataset element, which is a string of length 22.
    The first 20 characters are either "0" or "1", then there is
    a "|" character, then the answer, which is either "0" or "1".
    """
    # create a random binary string (with numeric values for now)
    bits = np.random.randint(0, 2, size=20)
    # compute the parity of the first 3 bits
    parity = np.sum(bits[:3]) % 2
    # convert to strings
    bits = "".join([str(bit) for bit in bits])
    return bits + "|" + str(parity)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    tokenizer = Tokenizer(tokenizer_vocabulary)

    D = int(1e5)
    split = 0.8

    strings = []
    answer_idxs = []
    for _ in tqdm(range(D)):
        full_string = create_string()
        strings.append(full_string)
        answer_idxs.append((len(tokenizer.encode(full_string)) - 1,))

    strings_train = strings[:int(split * D)]
    answer_idxs_train = answer_idxs[:int(split * D)]
    strings_test = strings[int(split * D):]
    answer_idxs_test = answer_idxs[int(split * D):]

    # save the dataset as a csv
    df = pd.DataFrame({"string": strings_train, "answer_idxs": answer_idxs_train})
    df.to_csv("train.csv", index=False)
    df = pd.DataFrame({"string": strings_test, "answer_idxs": answer_idxs_test})
    df.to_csv("test.csv", index=False)

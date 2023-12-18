
"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a binary classification task where the input is a
string of capital letters and the label is whether or not the string 
is a palindrome. After the string, there is a "|" character, 
followed by the label, which is either "y" or "n". For example, 
"ABC|n" is a negative example and "ABA|y" is a positive example. 
The network is evaluated on the last token of the string, which is 
the label.

Input strings of length 3 to 10. With the end of string "|" and the
label ("y" or "n"), the maximum length of the input is 12. We tokenize 
this string by character, so the maximum input sequence length to the
model is 12. String lengths are uniformly distributed between 3 and 10.
Approximately 50% of the strings are palindromes. To generate non-
palindromes, we generate a random string of the given length. To 
generate palindromes, we generate a random string of the given length
and then mirror the first half of the string to the second half.

Note that near-miss non-palindromes e.g. "ABCBAD|n" are unlikely to be 
generated. This could cause a model trained on this dataset to not be
robust to  to strings that are close to palindromes but are not 
palindromes.

-------------------------------------------------------------------
"""

import os
from math import floor, ceil
import random

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from neural_verification import Tokenizer
from config import tokenizer_vocabulary

def create_string(length, is_palindrome) -> str:
    if is_palindrome:
        characters = np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), size=length).tolist()
        return  "".join(characters[:ceil(length / 2)] + characters[:floor(length / 2)][::-1])
    else:
        candidate = "".join(np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), size=length).tolist())
        if candidate == candidate[::-1]: # re-run if accidentally a palindrome
            return create_string(length, is_palindrome)
        else:   
            return candidate

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    tokenizer = Tokenizer(tokenizer_vocabulary)

    D = int(1e5)
    split = 0.8

    strings = []
    answer_idxs = []
    for _ in tqdm(range(D)):
        is_palindrome = np.random.choice([True, False])
        length = np.random.randint(3, 10+1) # uniform on {3, ..., 10}
        label = "y" if is_palindrome else "n"
        full_string = create_string(length, is_palindrome) + "|" + label
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

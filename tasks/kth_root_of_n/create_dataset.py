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

    tokenizer = Tokenizer(tokenizer_vocabulary)

    split = 0.8

    strings = []
    answer_idxs = []
    n_values = np.arange(1, 10001)
    k_values = np.arange(2, 5)
    
    for i, n in enumerate(tqdm(n_values)):
        for j, k in enumerate(k_values):
            roots = np.round(n ** (1/k)).astype(int)
            full_string = f"{n},{k}|{roots}" 
            strings.append(full_string)
            answer_idxs.append((len(tokenizer.encode(full_string)) - 1,))

    
    df = pd.DataFrame({"string": strings, "answer_idxs": answer_idxs})

    # save the dataset as a csv
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    train_df.to_csv("train_transformer.csv", index=False)
    test_df.to_csv("test_transformer.csv", index=False)
    
from math import floor, ceil
import random

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from neural_verification import Tokenizer
from config import tokenizer_vocabulary, dummy_token

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    tokenizer = Tokenizer(tokenizer_vocabulary)

    split = 0.8

    inputs = []
    labels = []
    answer_idxs = []
    n_values = np.arange(1, 1000001)
    k_values = [2,3,5,6,8,9,10]
    for i, n in enumerate(tqdm(n_values)):
        for j, k in enumerate(k_values):
            roots = f"{np.round(n ** (1/k)).astype(int)}"
            ## generate a string of Xs matching the length of the roots
            dummy_tokens = "".join([dummy_token for _ in range(len(tokenizer.encode(roots)))])
            dummy_string = f"{n},{k}|{dummy_tokens}" 
            full_string = f"{n},{k}|{roots}" 
            inputs.append(dummy_string)
            labels.append(full_string)
            answer_idx_start = len(tokenizer.encode(full_string)) - len(tokenizer.encode(f"{roots}"))
            answer_idx_end = len(tokenizer.encode(full_string))
            answer_idx_element = (answer_idx_start, )
            for answer_idx in range(answer_idx_start+1, answer_idx_end):
                answer_idx_element += (answer_idx, )
            answer_idxs.append(answer_idx_element)
    
    df = pd.DataFrame({"inputs": inputs, "labels": labels, "answer_idxs": answer_idxs})

    # save the dataset as a csv
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    
    
    inputs = []
    labels = []
    answer_idxs = []
    k_values = [4, 7]
    for i, n in enumerate(tqdm(n_values)):
        for j, k in enumerate(k_values):
            roots = f"{np.round(n ** (1/k)).astype(int)}"
            ## generate a string of Xs matching the length of the roots
            dummy_tokens = "".join([dummy_token for _ in range(len(tokenizer.encode(roots)))])
            dummy_string = f"{n},{k}|{dummy_tokens}" 
            full_string = f"{n},{k}|{roots}" 
            inputs.append(dummy_string)
            labels.append(full_string)
            answer_idx_start = len(tokenizer.encode(full_string)) - len(tokenizer.encode(f"{roots}"))
            answer_idx_end = len(tokenizer.encode(full_string))
            answer_idx_element = (answer_idx_start, )
            for answer_idx in range(answer_idx_start+1, answer_idx_end):
                answer_idx_element += (answer_idx, )
            answer_idxs.append(answer_idx_element)
    
    additional_df = pd.DataFrame({"inputs": inputs, "labels": labels, "answer_idxs": answer_idxs})
    test_df = pd.concat([test_df, additional_df.sample(frac=0.5, random_state=42)])
    
    train_df.to_csv("tasks/kth_root_of_n_multidigit/train_funky.csv", index=False, quotechar='"', quoting=1)
    test_df.to_csv("tasks/kth_root_of_n_multidigit/test_funky.csv", index=False, quotechar='"', quoting=1)
    
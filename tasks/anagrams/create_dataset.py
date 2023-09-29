from math import floor, ceil
import random
import string
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from neural_verification import Tokenizer
from config import tokenizer_vocabulary

def create_string(length: int, is_anagram: bool) -> str:
    """
    Create two strings according to `is_anagram` and return the concatenation.
    
    Parameters:
    length (int): The length of each string to be generated.
    is_anagram (bool): A flag to determine if the two strings should be anagrams.
    
    Returns:
    str: The concatenated string.
    """
    if length <= 0:
        return ""
    
    # Generate the first string randomly
    first_string = ''.join(random.choice(string.ascii_uppercase) for _ in range(length))
    
    if is_anagram:
        # If the strings should be anagrams, shuffle the first string to create the second
        second_string_list = list(first_string)
        random.shuffle(second_string_list)
        second_string = ''.join(second_string_list)
    else:
        # If the strings should not be anagrams, generate a new random string for the second string
        second_string = ''.join(random.choice(string.ascii_uppercase) for _ in range(length))
    
    return first_string, second_string


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    tokenizer = Tokenizer(tokenizer_vocabulary)

    D = int(1e5)
    split = 0.8

    strings = []
    answer_idxs = []
    for _ in tqdm(range(D)):
        is_anagram = np.random.choice([True, False])
        length = np.random.randint(3, 10+1) # uniform on {3, ..., 10}
        label = "y" if is_anagram else "n"
        original, anagram = create_string(length, is_anagram)
        full_string = original + "," + anagram + "|" + label
        strings.append(full_string)
        answer_idxs.append((len(tokenizer.encode(full_string)) - 1,))

    strings_train = strings[:int(split * D)]
    answer_idxs_train = answer_idxs[:int(split * D)]
    strings_test = strings[int(split * D):]
    answer_idxs_test = answer_idxs[int(split * D):]

    # save the dataset as a csv
    df = pd.DataFrame({"string": strings_train, "answer_idxs": answer_idxs_train})
    df.to_csv("tasks/anagrams/train.csv", index=False)
    df = pd.DataFrame({"string": strings_test, "answer_idxs": answer_idxs_test})
    df.to_csv("tasks/anagrams/test.csv", index=False)

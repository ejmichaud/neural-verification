import numpy as np
import random

def generate_dataset(size):
    sequences = []
    for _ in range(size):
        sequence_length = random.randint(5, 10)  # arbitrary sequence length between 5 and 10
        sequence = sorted([random.randint(1, 9) for _ in range(sequence_length)])
        target = sorted(set(sequence)) # remove duplicates while preserving order
        sequences.append(("".join(map(str, sequence)), "".join(map(str, target))))
        # sequences.append((sequence, target))
    return sequences

# Example usage:
dataset = generate_dataset(100000)
# save 80% of the dataset array to data_train.txt and the rest to data_test.txt
np.savetxt("data_train.txt", dataset[:80000], fmt="%s")
np.savetxt("data_test.txt", dataset[80000:], fmt="%s")

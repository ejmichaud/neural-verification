import numpy as np
import random

def generate_dataset(size):
    sequences = []
    for _ in range(size):
        sequence_length = random.randint(5, 10)  # arbitrary sequence length between 5 and 10
        sequence = [random.randint(0, 9) for _ in range(sequence_length)]
        sequences.append(("".join(map(str, sequence)), str(sum(sequence))))
        # sequences.append((sequence, target))
    return sequences

# Example usage:
dataset = generate_dataset(100000)
# save 80% of the dataset array to data_train.txt and the rest to data_test.txt
np.savetxt("data_train.txt", dataset[:80000], fmt="%s")
np.savetxt("data_test.txt", dataset[80000:], fmt="%s")

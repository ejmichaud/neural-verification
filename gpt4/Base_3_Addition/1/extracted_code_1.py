import numpy as np

# Define the lists of x and y values
list_x = [
    [[0, 2], [1, 1], [0, 2], [1, 2], [1, 2], [1, 1], [2, 0], [0, 1], [2, 1], [0, 1]],
    [[1, 2], [1, 2], [2, 1], [2, 0], [1, 1], [0, 0], [1, 1], [2, 1], [2, 2], [1, 0]],
    [[2, 0], [1, 0], [0, 2], [2, 1], [0, 2], [0, 0], [1, 0], [1, 1], [1, 2], [2, 0]],
    [[2, 0], [2, 1], [0, 2], [0, 1], [1, 1], [2, 0], [1, 1], [1, 2], [0, 1], [2, 2]],
    [[0, 1], [2, 0], [1, 2], [1, 0], [2, 1], [1, 2], [0, 0], [1, 2], [1, 1], [2, 2]]
]

list_y = [
    [2, 2, 2, 0, 1, 0, 0, 2, 0, 2],
    [0, 1, 1, 0, 0, 1, 2, 0, 2, 2],
    [2, 1, 2, 0, 0, 1, 1, 2, 0, 0],
    [2, 0, 0, 2, 2, 2, 2, 0, 2, 1],
    [1, 2, 0, 2, 0, 1, 1, 0, 0, 2]
]

# Function to compute y based on a guessed rule
def compute_y(x):
    return sum(x) % 3

# Check the guessed rule against the actual values
for i in range(5):
    computed_y = [compute_y(pair) for pair in list_x[i]]
    if computed_y == list_y[i]:
        print(f"Row {i+1}: Success")
    else:
        print(f"Row {i+1}: Failure")
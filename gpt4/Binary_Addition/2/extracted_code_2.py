import numpy as np

# Define the list x and list y from the first five rows of the provided data.
list_x = [
    [0, 1], [0, 0], [0, 1], [0, 0], [0, 1], [0, 0], [0, 0], [1, 0], [1, 1], [1, 0],
    [1, 0], [1, 1], [1, 1], [1, 1], [1, 1], [0, 0], [1, 1], [1, 0], [1, 0], [0, 0],
    [0, 0], [1, 1], [1, 1], [1, 0], [1, 1], [0, 1], [0, 1], [0, 1], [1, 0], [0, 0],
    [0, 0], [0, 0], [0, 1], [1, 0], [1, 1], [1, 1], [0, 1], [0, 1], [1, 1], [0, 1],
    [0, 1], [0, 1], [0, 0], [1, 0], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 0]
]

list_y = [
    [1, 0, 1, 0, 1, 0, 0, 1, 0, 0],
    [1, 0, 1, 1, 1, 1, 0, 0, 0, 1],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 1, 0, 0, 1, 0],
    [1, 1, 0, 1, 0, 1, 1, 1, 1, 0]
]

# A function to compute list y from list x, using a heuristic rule.
def compute_y_from_x(list_x):
    computed_y = []
    # Attempting a heuristic: counting how many times [1, 1] appears within a group of list_x elements
    for i, x in enumerate(list_x):
        if i % 10 == 0:
            count_ones = sum(1 for j in range(i, i + 10) if list_x[j] == [1, 1])
            # Mock heuristic rule: if count of [1, 1] is even, then y = 0 else y = 1
            computed_y.append(1 if count_ones % 2 else 0)
    return computed_y

# Checking if the computed y matches the provided y for each row.
def check_success(list_x, list_y):
    for i in range(5):
        computed_y = compute_y_from_x(list_x[i*10:(i+1)*10])
        if computed_y == list_y[i]:
            print(f"Row {i + 1}: Success")
        else:
            print(f"Row {i + 1}: Failure")
            print(f"Computed y: {computed_y}")
            print(f"Provided y: {list_y[i]}")

# Run the check.
check_success(list_x, list_y)
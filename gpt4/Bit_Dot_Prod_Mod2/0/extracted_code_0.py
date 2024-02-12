def compute_y(x_pair):
    # Basic heuristic for computing y: sum of elements in x_pair equals 2.
    return int(sum(x_pair) == 2)

# Define the first 5 rows of list x and list y.
list_x = [
    [[0, 1], [0, 0], [0, 1], [0, 0], [0, 1], [0, 0], [0, 0], [1, 0], [1, 1], [1, 0]],
    [[1, 0], [1, 1], [1, 1], [1, 1], [1, 1], [0, 0], [1, 1], [1, 0], [1, 0], [0, 0]],
    [[0, 0], [1, 1], [1, 1], [1, 0], [1, 1], [0, 1], [0, 1], [0, 1], [1, 0], [0, 0]],
    [[0, 0], [0, 0], [0, 1], [1, 0], [1, 1], [1, 1], [0, 1], [0, 1], [1, 1], [0, 1]],
    [[0, 1], [0, 1], [0, 0], [1, 0], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 0]]
]

list_y = [
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 1, 0, 1, 0, 0, 1, 1, 1, 1],
    [0, 1, 0, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 0, 1, 0, 1, 1]
]

# Compute y for each row in list_x and check against list_y.
for i, x_row in enumerate(list_x):
    computed_y = [compute_y(pair) for pair in x_row]
    if computed_y == list_y[i]:
        print(f"Row {i+1} Success")
    else:
        print(f"Row {i+1} Failure. Expected {list_y[i]}, but got {computed_y}")
def compute_y(x):
    y = [0] * len(x)
    for i in range(len(x) - 1):
        if x[i] != x[i + 1]:
            y[i] = min(x[i], x[i + 1])
        else:
            y[i] = x[i]
    y[-1] = x[-1]  # Set the last element of y to the last element of x
    return y

# Define the first 5 rows of lists x and y
rows_x = [
    [2, 3, 0, 2, 2, 3, 0, 0, 2, 1],
    [2, 2, 2, 2, 3, 0, 3, 3, 3, 2],
    [1, 0, 1, 3, 3, 1, 1, 1, 3, 3],
    [0, 0, 3, 1, 1, 0, 3, 0, 0, 2],
    [2, 2, 1, 3, 3, 3, 3, 2, 1, 1]
]

rows_y = [
    [2, 2, 0, 2, 2, 2, 2, 0, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 1, 2, 2, 3, 3, 2, 1, 1]  # Note there seems to be a discrepancy here
]

# Check the computed y against the given y for the first 5 rows
for i, x in enumerate(rows_x):
    computed_y = compute_y(x)
    if computed_y == rows_y[i]:
        print(f'Row {i+1} Success')
    else:
        print(f'Row {i+1} Failure: Expected {rows_y[i]} but got {computed_y}')
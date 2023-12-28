def compute_y(x):
    # Hypothetical function to calculate y from x
    return [a ^ b for a, b in x]

# Define the input x and the expected y for the first 5 rows
rows_x = [
    [[0, 1], [0, 0], [0, 1], [0, 0], [0, 1], [0, 0], [0, 0], [1, 0], [1, 1], [1, 0]],
    [[1, 0], [1, 1], [1, 1], [1, 1], [1, 1], [0, 0], [1, 1], [1, 0], [1, 0], [0, 0]],
    [[0, 0], [1, 1], [1, 1], [1, 0], [1, 1], [0, 1], [0, 1], [0, 1], [1, 0], [0, 0]],
    [[0, 0], [0, 0], [0, 1], [1, 0], [1, 1], [1, 1], [0, 1], [0, 1], [1, 1], [0, 1]],
    [[0, 1], [0, 1], [0, 0], [1, 0], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 0]]
]

rows_y = [
    [1, 0, 1, 0, 1, 0, 0, 1, 0, 0],
    [1, 0, 1, 1, 1, 1, 0, 0, 0, 1],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 1, 0, 0, 1, 0],
    [1, 1, 0, 1, 0, 1, 1, 1, 1, 0]
]

# Check each row to see if the computed y matches the given y
for i, x in enumerate(rows_x):
    computed_y = compute_y(x)
    if computed_y == rows_y[i]:
        print(f"Row {i+1} Success: {computed_y} matches {rows_y[i]}")
    else:
        print(f"Row {i+1} Failure: {computed_y} does not match {rows_y[i]}")
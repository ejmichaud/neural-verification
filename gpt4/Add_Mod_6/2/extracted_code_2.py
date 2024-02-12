def compute_y(x):
    y = []
    for num in x:
        if num in [0, 1]:
            y.append(num)
        elif num == 5:
            y.append(num - 1)
        elif num % 2 == 0:
            y.append(num - 1)
        else:
            # Unable to determine a pattern so far.
            # We need a default prediction which will likely be wrong.
            y.append(None)
    return y

# Provided data
rows = [
    ([0, 5, 4, 4, 0, 5, 4, 2, 4, 5], [0, 5, 3, 1, 1, 0, 4, 0, 4, 3]),
    ([4, 4, 2, 0, 3, 4, 5, 1, 3, 4], [4, 2, 4, 4, 1, 5, 4, 5, 2, 0]),
    ([1, 2, 1, 5, 5, 1, 5, 3, 1, 1], [1, 3, 4, 3, 2, 3, 2, 5, 0, 1]),
    ([0, 0, 1, 1, 5, 4, 5, 2, 4, 0], [0, 0, 1, 2, 1, 5, 4, 0, 4, 4]),
    ([2, 0, 1, 3, 3, 5, 5, 4, 3, 5], [2, 2, 3, 0, 3, 2, 1, 5, 2, 1])
]

# Test the compute_y function for the first 5 rows
for i, (x_list, y_list) in enumerate(rows):
    computed_y = compute_y(x_list)
    if computed_y == y_list:
        print(f"Row {i+1} Success: {computed_y}")
    else:
        print(f"Row {i+1} Failure: Computed: {computed_y}, Expected: {y_list}")
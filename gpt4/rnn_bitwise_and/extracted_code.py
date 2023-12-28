def f(pair):
    """Function to determine the value in list y based on the pair in list x."""
    return 1 if pair == [1, 1] else 0

# Given lists x for the first 5 rows
lists_x = [
    [[0, 1],[0, 0],[0, 1],[0, 0],[0, 1],[0, 0],[0, 0],[1, 0],[1, 1],[1, 0]],
    [[1, 0],[1, 1],[1, 1],[1, 1],[1, 1],[0, 0],[1, 1],[1, 0],[1, 0],[0, 0]],
    [[0, 0],[1, 1],[1, 1],[1, 0],[1, 1],[0, 1],[0, 1],[0, 1],[1, 0],[0, 0]],
    [[0, 0],[0, 0],[0, 1],[1, 0],[1, 1],[1, 1],[0, 1],[0, 1],[1, 1],[0, 1]],
    [[0, 1],[0, 1],[0, 0],[1, 0],[1, 1],[1, 1],[1, 1],[1, 1],[1, 1],[1, 0]]
]

# Given lists y for the first 5 rows
lists_y = [
    [0,0,0,0,0,0,0,0,1,0],
    [0,1,1,1,1,0,1,0,0,0],
    [0,1,1,0,1,0,0,0,0,0],
    [0,0,0,0,1,1,0,0,1,0],
    [0,0,0,0,1,1,1,1,1,0]
]

# Compute list y from list x and check against the given list y
for index, list_x in enumerate(lists_x):
    computed_y = [f(pair) for pair in list_x]
    if computed_y == lists_y[index]:
        print(f"Row {index + 1}: Success")
    else:
        print(f"Row {index + 1}: Failure")
        print(f"Expected: {lists_y[index]}")
        print(f"Computed: {computed_y}")
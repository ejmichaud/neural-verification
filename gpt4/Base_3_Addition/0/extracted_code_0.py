def compute_y(x_pair):
    # multiply the elements and take modulo 3
    return (x_pair[0] * x_pair[1]) % 3

# Given lists x and y
lists_x = [
    [[0, 2],[1, 1],[0, 2],[1, 2],[1, 2],[1, 1],[2, 0],[0, 1],[2, 1],[0, 1]],
    [[1, 2],[1, 2],[2, 1],[2, 0],[1, 1],[0, 0],[1, 1],[2, 1],[2, 2],[1, 0]],
    [[2, 0],[1, 0],[0, 2],[2, 1],[0, 2],[0, 0],[1, 0],[1, 1],[1, 2],[2, 0]],
    [[2, 0],[2, 1],[0, 2],[0, 1],[1, 1],[2, 0],[1, 1],[1, 2],[0, 1],[2, 2]],
    [[0, 1],[2, 0],[1, 2],[1, 0],[2, 1],[1, 2],[0, 0],[1, 2],[1, 1],[2, 2]]
]

lists_y = [
    [2, 2, 2, 0, 1, 0, 0, 2, 0, 2],
    [0, 1, 1, 0, 0, 1, 2, 0, 2, 2],
    [2, 1, 2, 0, 0, 1, 1, 2, 0, 0],
    [2, 0, 0, 2, 2, 2, 2, 0, 2, 1],
    [1, 2, 0, 2, 0, 1, 1, 0, 0, 2]
]

# Compute and check list y for each row
for i in range(5):  # iterating through the first 5 rows
    computed_y = [compute_y(pair) for pair in lists_x[i]]  # using the hypothesized function
    if computed_y == lists_y[i]:
        print(f"Row {i+1} Success")
    else:
        print(f"Row {i+1} Failure")
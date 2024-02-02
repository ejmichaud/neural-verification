def compute_y(x, mapping):
    return [mapping.get(tuple(pair), "unknown") for pair in x]

# Define a hardcoded mapping based on the first five rows (this is just a dummy mapping)
mapping = {
    (0, 1): 1,
    (0, 0): 0,
    (1, 0): 1,
    (1, 1): 0,
}

# The example list x's (first five rows)
list_xs = [
    [[0, 1],[0, 0],[0, 1],[0, 0],[0, 1],[0, 0],[0, 0],[1, 0],[1, 1],[1, 0]],
    [[1, 0],[1, 1],[1, 1],[1, 1],[1, 1],[0, 0],[1, 1],[1, 0],[1, 0],[0, 0]],
    [[0, 0],[1, 1],[1, 1],[1, 0],[1, 1],[0, 1],[0, 1],[0, 1],[1, 0],[0, 0]],
    [[0, 0],[0, 0],[0, 1],[1, 0],[1, 1],[1, 1],[0, 1],[0, 1],[1, 1],[0, 1]],
    [[0, 1],[0, 1],[0, 0],[1, 0],[1, 1],[1, 1],[1, 1],[1, 1],[1, 1],[1, 0]],
]

# The example list y's (corresponding to the first five rows)
list_ys = [
    [1,0,1,0,1,0,0,1,0,0],
    [1,0,1,1,1,1,0,0,0,1],
    [0,0,1,0,1,0,0,0,0,1],
    [0,0,1,1,0,1,0,0,1,0],
    [1,1,0,1,0,1,1,1,1,0],
]

# Compute and check list y's from list x's
for i in range(len(list_xs)):
    computed_y = compute_y(list_xs[i], mapping)
    if computed_y == list_ys[i]:
        print(f"Row {i+1} -- Success")
    else:
        print(f"Row {i+1} -- Failure")
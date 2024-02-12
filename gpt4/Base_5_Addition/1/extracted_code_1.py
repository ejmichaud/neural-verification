def compute_y(x_pair):
    # This is a placeholder function. Since we do not know the actual
    # relationship between x and y, we'll try an arbitrary operation.
    # In this case, we are subtracting the second element from the first.
    return x_pair[0] - x_pair[1]

# The first five rows of list x
x_lists = [
    [2, 2],[1, 4],[1, 0],[0, 4],[0, 3],[3, 4],[0, 4],[1, 2],[0, 0],[2, 1],
    [4, 1],[3, 1],[4, 3],[1, 4],[2, 4],[2, 0],[0, 4],[3, 4],[4, 1],[2, 0],
    [1, 2],[2, 4],[2, 3],[3, 4],[3, 2],[0, 4],[0, 4],[1, 4],[0, 4],[3, 3],
    [1, 0],[0, 0],[0, 1],[3, 0],[1, 1],[2, 4],[4, 3],[3, 4],[3, 2],[3, 1],
    [4, 1],[1, 3],[2, 0],[4, 3],[0, 3],[2, 2],[0, 4],[1, 0],[1, 4],[1, 4],
]

# The actual y values for the first five rows
y_lists = [
    4,0,2,4,3,2,0,4,0,3,
    0,0,3,1,2,3,4,2,1,3,
    3,1,1,3,1,0,0,1,0,2,
    1,0,1,3,2,1,3,3,1,0,
    0,0,3,2,4,4,4,1,0,1,
]

# We're going to check each row in x_lists to see if our compute_y function matches the y_lists
for i, x_list in enumerate(x_lists):
    y_computed = [compute_y(x_pair) for x_pair in zip(*[iter(x_list)]*2)]
    if y_computed == y_lists[i*10:(i+1)*10]:
        print(f"Row {i+1} - Success")
    else:
        print(f"Row {i+1} - Failure: Expected {y_lists[i*10:(i+1)*10]}, got {y_computed}")
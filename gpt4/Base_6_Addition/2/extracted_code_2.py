def compute_y(x):
    # This is a placeholder function and needs to be replaced with the actual relationship between x and y
    # Here we're simply returning the difference between the two elements
    return abs(x[0] - x[1])

# Define the list x and list y for the first 5 rows
input_lists_x = [
    [0, 5],[4, 4],[0, 5],[4, 2],[4, 5],[4, 4],[2, 0],[3, 4],[5, 1],[3, 4],
    [1, 2],[1, 5],[5, 1],[5, 3],[1, 1],[0, 0],[1, 1],[5, 4],[5, 2],[4, 0],
    [2, 0],[1, 3],[3, 5],[5, 4],[3, 5],[0, 3],[4, 3],[4, 1],[1, 2],[2, 0],
    [2, 0],[2, 4],[0, 5],[3, 4],[1, 1],[5, 3],[4, 1],[4, 5],[3, 1],[2, 5],
    [0, 1],[2, 3],[4, 2],[1, 0],[5, 1],[1, 5],[3, 3],[1, 5],[1, 1],[5, 2]
]

expected_lists_y = [
    5,2,0,1,4,3,3,1,1,2,
    3,0,1,3,3,0,2,3,2,5,
    2,4,2,4,3,4,1,0,4,2,
    2,0,0,2,3,2,0,4,5,1,
    1,5,0,2,0,1,1,1,3,1
]

# Compute y for each list in x and compare with the expected y
for i, list_x in enumerate(input_lists_x):
    computed_y = [compute_y(x) for x in zip(list_x[::2], list_x[1::2])]
    print(f"Row {i+1} - Computed y: {computed_y}, Expected y: {expected_lists_y[i*10:(i+1)*10]}")
    if computed_y == expected_lists_y[i*10:(i+1)*10]:
        print("Success")
    else:
        print("Failure")
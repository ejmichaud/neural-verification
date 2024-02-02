def compute_y(x):
    y = x[:]
    # Only applying a known rule inferred from the first element observation
    # y[0] = x[0]
    # For the rest of the elements, we will keep them same for now,
    # since we do not have a clear rule for the transformation. 
    return y

# List x and List y as provided in the question for the first 5 rows
list_x = [
    [2, 2, 1, 4, 1, 0, 0, 4, 0, 3],
    [3, 4, 0, 4, 1, 2, 0, 0, 2, 1],
    [4, 1, 3, 1, 4, 3, 1, 4, 2, 4],
    [2, 0, 0, 4, 3, 4, 4, 1, 2, 0],
    [1, 2, 2, 4, 2, 3, 3, 4, 3, 2],
]

list_y = [
    [2, 4, 0, 4, 0, 0, 0, 4, 4, 2],
    [3, 2, 2, 1, 2, 4, 4, 4, 1, 2],
    [4, 0, 3, 4, 3, 1, 2, 1, 3, 2],
    [2, 2, 2, 1, 4, 3, 2, 3, 0, 0],
    [1, 3, 0, 4, 1, 4, 2, 1, 4, 1],
]

# Compute y from x and check if it matches the given y
for i in range(len(list_x)):
    computed_y = compute_y(list_x[i])
    if computed_y == list_y[i]:
        print(f'Success [Row {i+1}]')
    else:
        print(f'Failure [Row {i+1}]')
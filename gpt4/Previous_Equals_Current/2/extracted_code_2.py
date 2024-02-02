def compute_y_from_x(x):
    y = []
    for i in range(len(x)):
        if x[i] == 7:
            # y is 1 if it's the first element in the list or not following another 7
            y.append(1 if i == 0 or x[i-1] != 7 else 0)
        else:
            # y is 1 if the current number follows a smaller number, but not if it or the previous one is a 7
            y.append(1 if (i != 0 and x[i] > x[i-1] and x[i-1] != 7) else 0)
    return y

# Given lists
lists_x = [
    [0, 7, 7, 4, 7, 7, 2, 2, 6, 0],
    [0, 6, 7, 5, 4, 4, 2, 6, 3, 0],
    [6, 7, 1, 5, 3, 3, 1, 6, 3, 3],
    [3, 6, 0, 6, 5, 6, 4, 7, 6, 6],
    [6, 0, 3, 5, 3, 3, 1, 3, 1, 3]
]
lists_y = [
    [1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
]

# Process each row and check if output matches expected y
for i in range(len(lists_x)):
    computed_y = compute_y_from_x(lists_x[i])
    if computed_y == lists_y[i]:
        print(f'Row {i+1}: Success')
    else:
        print(f'Row {i+1}: Failure')
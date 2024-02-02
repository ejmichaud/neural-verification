def compute_y(x_list):
    y_list = []
    for i in range(len(x_list)):
        # Check if current pair is (1,1)
        if x_list[i] == (1, 1):
            # Check if it's the first element or the previous pair was (0,0)
            if i == 0 or x_list[i-1] == (0, 0):
                y_list.append(1)
            # If the previous is not (1,1) and one before the previous pair is (0,0)
            elif i > 1 and x_list[i-1] != (1, 1) and x_list[i-2] == (0, 0):
                y_list.append(1)
            else:
                y_list.append(0)
        else:
            y_list.append(0)
    return y_list

# Define the first 5 rows of list x and list y
x_rows = [
    [0, 1],[0, 0],[0, 1],[0, 0],[0, 1],[0, 0],[0, 0],[1, 0],[1, 1],[1, 0],  # Row 1
    [1, 0],[1, 1],[1, 1],[1, 1],[1, 1],[0, 0],[1, 1],[1, 0],[1, 0],[0, 0],  # Row 2
    [0, 0],[1, 1],[1, 1],[1, 0],[1, 1],[0, 1],[0, 1],[0, 1],[1, 0],[0, 0],  # Row 3
    [0, 0],[0, 0],[0, 1],[1, 0],[1, 1],[1, 1],[0, 1],[0, 1],[1, 1],[0, 1],  # Row 4
    [0, 1],[0, 1],[0, 0],[1, 0],[1, 1],[1, 1],[1, 1],[1, 1],[1, 1],[1, 0],  # Row 5
]

y_rows = [
    [0,0,0,0,0,0,0,0,1,0],
    [0,1,1,1,1,0,1,0,0,0],
    [0,1,1,0,1,0,0,0,0,0],
    [0,0,0,0,1,1,0,0,1,0],
    [0,0,0,0,1,1,1,1,1,0],
]

# Check if the computed list y matches the given list y
for i, x in enumerate(x_rows):
    y_computed = compute_y([tuple(x[j:j+2]) for j in range(0, len(x), 2)])
    if y_computed == y_rows[i]:
        print(f'Row {i+1} Success')
    else:
        print(f'Row {i+1} Failure')
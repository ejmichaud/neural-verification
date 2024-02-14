def compute_y(x_list):
    # This function returns the second element of each pair in x_list.
    # This is an arbitrary assumption for demonstration purposes.
    y_computed = [x[1] for x in x_list]
    return y_computed

# Define the first five rows of list x and list y as provided in the prompt
list_x = [
    [[0, 1],[0, 0],[0, 1],[0, 0],[0, 1],[0, 0],[0, 0],[1, 0],[1, 1],[1, 0]],
    [[1, 0],[1, 1],[1, 1],[1, 1],[1, 1],[0, 0],[1, 1],[1, 0],[1, 0],[0, 0]],
    [[0, 0],[1, 1],[1, 1],[1, 0],[1, 1],[0, 1],[0, 1],[0, 1],[1, 0],[0, 0]],
    [[0, 0],[0, 0],[0, 1],[1, 0],[1, 1],[1, 1],[0, 1],[0, 1],[1, 1],[0, 1]],
    [[0, 1],[0, 1],[0, 0],[1, 0],[1, 1],[1, 1],[1, 1],[1, 1],[1, 1],[1, 0]]
]
list_y = [
    [1,0,1,0,1,0,0,1,0,1],
    [1,0,0,0,0,0,0,1,1,0],
    [0,0,0,1,0,1,1,1,1,0],
    [0,0,1,1,0,0,1,1,0,1],
    [1,1,0,1,0,0,0,0,0,1]
]

# Check to see if the computed y matches the actual y for all rows
for index, x in enumerate(list_x):
    y_computed = compute_y(x)
    y_actual = list_y[index]
    if y_computed == y_actual:
        print(f"Row {index + 1} Success")
    else:
        print(f"Row {index + 1} Failure")
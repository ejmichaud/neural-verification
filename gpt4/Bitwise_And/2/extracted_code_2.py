def compute_y(x):
    # Initial condition for the first element
    y = [0] * len(x)
    
    # Starting from the second pair, we need to consider the pair before
    for i in range(1, len(x)):
        if x[i] == [1, 1]:
            y[i] = 1
        elif x[i] == [1, 0] and x[i-1] == [0, 1]:
            y[i] = 1
        else:
            y[i] = 0
          
    # Check the first element
    if x[0] == [1, 1]:
        y[0] = 1
    
    return y

# List x from the first 5 rows
list_x = [
    [[0, 1],[0, 0],[0, 1],[0, 0],[0, 1],[0, 0],[0, 0],[1, 0],[1, 1],[1, 0]],
    [[1, 0],[1, 1],[1, 1],[1, 1],[1, 1],[0, 0],[1, 1],[1, 0],[1, 0],[0, 0]],
    [[0, 0],[1, 1],[1, 1],[1, 0],[1, 1],[0, 1],[0, 1],[0, 1],[1, 0],[0, 0]],
    [[0, 0],[0, 0],[0, 1],[1, 0],[1, 1],[1, 1],[0, 1],[0, 1],[1, 1],[0, 1]],
    [[0, 1],[0, 1],[0, 0],[1, 0],[1, 1],[1, 1],[1, 1],[1, 1],[1, 1],[1, 0]]
]
# List y for the first 5 rows
list_y = [
    [0,0,0,0,0,0,0,0,1,0],
    [0,1,1,1,1,0,1,0,0,0],
    [0,1,1,0,1,0,0,0,0,0],
    [0,0,0,0,1,1,0,0,1,0],
    [0,0,0,0,1,1,1,1,1,0]
]

# Process each row and check if it matches list y
for i in range(5):
    computed_y = compute_y(list_x[i])
    if computed_y == list_y[i]:
        print(f"Row {i+1} Success")
    else:
        print(f"Row {i+1} Failure")
def compute_y_list(x_list):
    # Hypothesis function to compute a y value based on list x
    def hypothesis_function(x_previous, x_current, y_previous):
        # Rule 1: If we transition from [0, 0] to [1, *] or from [1, 0] to [1, 1], then y=1
        if (x_previous == [0, 0] and x_current[0] == 1) or (x_previous == [1, 0] and x_current == [1, 1]):
            return 1
        # Rule 2: If none of the above, then y=0
        else:
            return 0

    y_computed = []
    x_previous = None
    y_previous = None
    
    for x_current in x_list:
        if x_previous is None:  # First element has no previous
            y_current = 0
        else:
            y_current = hypothesis_function(x_previous, x_current, y_previous)
            
        y_computed.append(y_current)
        x_previous = x_current
        y_previous = y_current

    return y_computed

# The given lists of x
x_lists = [
    [[0, 1],[0, 0],[0, 1],[0, 0],[0, 1],[0, 0],[0, 0],[1, 0],[1, 1],[1, 0]],
    [[1, 0],[1, 1],[1, 1],[1, 1],[1, 1],[0, 0],[1, 1],[1, 0],[1, 0],[0, 0]],
    [[0, 0],[1, 1],[1, 1],[1, 0],[1, 1],[0, 1],[0, 1],[0, 1],[1, 0],[0, 0]],
    [[0, 0],[0, 0],[0, 1],[1, 0],[1, 1],[1, 1],[0, 1],[0, 1],[1, 1],[0, 1]],
    [[0, 1],[0, 1],[0, 0],[1, 0],[1, 1],[1, 1],[1, 1],[1, 1],[1, 1],[1, 0]]
]

# The given lists of y
y_lists = [
    [0,0,0,0,0,0,0,0,1,1],
    [0,1,0,1,0,0,1,1,1,1],
    [0,1,0,0,1,1,1,1,1,1],
    [0,0,0,0,1,0,0,0,1,1],
    [0,0,0,0,1,0,1,0,1,1]
]

# Perform the computations and print results
for i in range(len(x_lists)):
    y_computed = compute_y_list(x_lists[i])
    if y_computed == y_lists[i]:
        print(f'Row {i+1} Success')
    else:
        print(f'Row {i+1} Failure')
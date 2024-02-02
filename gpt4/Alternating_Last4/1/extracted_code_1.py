def compute_y(x):
    y = [0] * len(x)  # Initialize y with zeros
    
    # This is a placeholder for the real logic, which we need to deduce.
    # Let's start with the rule observed in hypothesis 1 above:
    # y[i] = 1 if x[i-2] == 1, x[i-1] == 0, x[i] == 1, x[i+1] == 0
    # Note: We also need to make sure we are not checking indices that don't exist.
    for i in range(2, len(x) - 2):  # we start at 2 and end at len(x) - 2 to avoid index errors
        if x[i-2] == 1 and x[i-1] == 0 and x[i] == 1 and x[i+1] == 0:
            y[i] = 1
    
    return y

# Given lists
list_xs = [
    [0,0,1,0,0,0,1,0,0,0],
    [1,1,1,0,1,0,1,1,1,1],
    [1,0,0,1,1,1,0,1,0,0],
    [1,1,1,1,1,0,1,1,0,1],
    [1,1,0,0,0,0,0,0,0,0]
]

list_ys = [
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,1,0,0,0],
    [0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0]
]

# Test the function for the first 5 rows
for i in range(5):
    result_y = compute_y(list_xs[i])
    if result_y == list_ys[i]:
        print('Success')
    else:
        print('Failure')
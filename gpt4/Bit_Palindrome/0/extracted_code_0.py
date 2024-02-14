# Define the function to compute y from x
def compute_y(x):
    n = len(x)
    y = [0] * n  # Initialize y with zeroes

    # Rule 1: if x is all zeros
    if all(v == 0 for v in x):
        return [1] * n
    
    # Initialize the first element of y
    y[0] = 1 if x[0] == 0 else x[0]

    # Apply the transformation rules for the rest of the elements
    for i in range(1, n):
        # Rule 2: if the i-th element of x is 1 and (i+1)-th is 0, then (i+1)-th of y is 0
        # Rule 3: if the i-th element of x is 0 and all elements before x[i] are 0, then y[i] is 1
        if x[i] == 0 and all(v == 0 for v in x[:i]):
            y[i] = 1
        else:
            y[i] = 0 if (x[i] == 1 and i+1 < n and x[i+1] == 0) else x[i]
    
    return y

# Given pairs of x and y lists for the first five rows
list_pairs = [
    ([1,0,0,0,0,0,0,1,0,1], [1,0,0,0,0,0,0,1,0,0]),
    ([0,0,0,0,0,0,0,1,0,0], [1,1,1,1,1,1,1,0,0,0]),
    ([1,1,1,1,1,1,0,1,0,0], [1,1,1,1,1,1,0,0,0,0]),
    ([1,1,1,0,0,1,0,1,1,1], [1,1,1,0,0,0,0,0,0,0]),
    ([1,1,1,0,1,1,1,0,1,0], [1,1,1,0,0,0,1,0,0,0])
]

# Process each pair and print 'Success' or 'Failure'
for x, expected_y in list_pairs:
    computed_y = compute_y(x)
    if computed_y == expected_y:
        print('Success')
    else:
        print('Failure')
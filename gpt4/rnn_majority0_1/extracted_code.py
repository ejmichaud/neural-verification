def compute_y(x):
    # Initialize y with all zeros
    y = [0] * len(x)
    # Start from the second element since the first is always going to be 0 in y
    for i in range(1, len(x)):
        # Apply the specified AND operation
        y[i] = x[i] & x[i-1]
    return y

# Given pairs of x and y lists
pairs = [
    ([0,1,0,0,0,1,0,0,0,1], [0,0,0,0,0,0,0,0,0,0]),
    ([0,0,0,0,1,0,1,1,1,0], [0,0,0,0,0,0,0,0,0,0]),
    ([1,0,1,1,1,1,1,1,1,1], [1,0,1,1,1,1,1,1,1,1]),
    ([0,0,1,1,1,0,1,0,0,0], [0,0,0,0,1,0,1,0,0,0]),
    ([0,0,1,1,1,1,1,0,1,1], [0,0,0,0,1,1,1,1,1,1])
]

# Iterate through the first five pairs
for x, expected_y in pairs[:5]:
    # Compute y from x
    computed_y = compute_y(x)
    # Compare the computed y with the expected y
    if computed_y == expected_y:
        print('Success')
    else:
        print('Failure')
        print(f'Computed y: {computed_y}')
        print(f'Expected y: {expected_y}')
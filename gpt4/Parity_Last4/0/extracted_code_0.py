def compute_y(x):
    y = [0] * len(x)  # Initialize list y with zeros
    for i in range(1, len(x)):
        if x[i] == 0:
            y[i] = y[i-1] if x[i-1] == 1 else 0
        else:
            y[i] = 0
    y[0] = 1 if x[0] == 0 else 0  # Initial condition for the first element
    return y

# Define the input-output pairs from the first five rows given
test_data = [
    ([0,1,0,0,0,1,0,0,0,1], [0,1,1,1,1,1,1,1,1,1]),
    ([0,0,0,0,1,0,1,1,1,0], [0,0,0,0,1,1,0,1,1,1]),
    ([1,0,1,1,1,1,1,1,1,1], [1,1,0,1,1,0,0,0,0,0]),
    ([0,0,1,1,1,0,1,0,0,0], [0,0,1,0,1,1,1,0,1,1]),
    ([0,0,1,1,1,1,1,0,1,1], [0,0,1,0,1,0,0,1,1,1]),
]

# Check each pair
for x_list, expected_y_list in test_data:
    # Compute y from x
    computed_y_list = compute_y(x_list)
    # Check if the computed y matches the expected y
    if computed_y_list == expected_y_list:
        print('Success')
    else:
        print('Failure')
        print(f"x: {x_list}")
        print(f"Expected y: {expected_y_list}")
        print(f"Computed y: {computed_y_list}")
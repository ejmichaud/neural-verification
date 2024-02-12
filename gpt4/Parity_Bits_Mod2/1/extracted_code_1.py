# The lists provided
list_x = [
    [0,1,0,0,0,1,0,0,0,1],
    [0,0,0,0,1,0,1,1,1,0],
    [1,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,0,1,0,0,0],
    [0,0,1,1,1,1,1,0,1,1]
]

# Corresponding list y values to check against
expected_list_y = [
    [0,1,0,1,0,1,0,1,0,1],
    [0,1,0,1,0,1,0,1,0,1],
    [0,1,0,1,0,1,0,1,0,1],
    [0,1,0,1,0,1,0,1,0,1],
    [0,1,0,1,0,1,0,1,0,1]
]

def compute_y_from_x(x_list):
    # We can disregard the contents of x since the pattern of y is fixed
    return [i % 2 for i in range(len(x_list))]

# Check each corresponding pair of lists
for i, x in enumerate(list_x):
    actual_y = compute_y_from_x(x)
    if actual_y == expected_list_y[i]:
        print(f"Row {i+1}: Success")
    else:
        print(f"Row {i+1}: Failure")
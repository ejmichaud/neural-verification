def compute_y_from_x(x):
    # Initialize list y with zeros
    y = [0] * len(x)
    
    # Check for '1's in list x by the hypothesis
    for i in range(1, len(x)-1):  # Skip the first and last element, already assumed to be 0
        if x[i-1] != x[i+1]:  # Check if '1' in x is followed or preceded by a '0', but not both
            y[i] = x[i]

    return y

# Define a function to check if the transformation is successful
def check_transformation(x, expected_y):
    computed_y = compute_y_from_x(x)
    if computed_y == expected_y:
        return 'Success'
    else:
        return 'Failure'

# List of test cases (first 5 rows from the provided data)
test_cases = [
    ([0,0,0,1,0,0,0,1,0,0], [0,0,0,0,1,0,0,0,1,0]),
    ([1,0,1,1,1,0,1,0,1,1], [0,1,1,0,0,0,1,1,1,0]),
    ([1,1,1,1,0,0,1,1,1,0], [0,0,0,0,0,0,0,0,0,0]),
    ([0,0,0,0,1,1,1,1,1,0], [0,0,0,0,0,0,0,0,0,0]),
    ([0,1,0,1,0,1,1,0,0,0], [0,0,1,1,1,1,0,0,0,0])
]

# Apply the transform and check each test case
for x, expected_y in test_cases:
    result = check_transformation(x, expected_y)
    print(result)